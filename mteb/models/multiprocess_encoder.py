from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from multiprocessing.connection import wait
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from multiprocessing.connection import Connection

    from torch.utils.data import DataLoader

    from mteb.models.model_meta import ModelMeta
    from mteb.types import Array, BatchedInput


@dataclass(frozen=True)
class MultiProcessEncoderConfig:
    """Execution settings for an encoder with one model replica per device.

    This configuration describes execution only: it must not be included in
    model experiment kwargs or embedding cache identities. Constructing it does
    not load a model, initialize devices, or start workers.

    Args:
        devices: Explicit worker devices, for example ``["cuda:0", "cuda:1"]``.
            Each entry creates one worker. Repeated ``"cpu"`` entries can be used
            for CPU workers. Device availability is checked when workers start,
            rather than when constructing this configuration.
        chunk_size: Number of inputs dispatched as one worker job. This is
            independent of ``encode_kwargs["batch_size"]``, which controls the
            inference batch size within each worker. Defaults to 2048.

    Examples:
        >>> config = MultiProcessEncoderConfig(devices=["cuda:0", "cuda:1"])
        >>> config.devices
        ('cuda:0', 'cuda:1')
    """

    devices: Sequence[str]
    chunk_size: int = 2048

    def __post_init__(self) -> None:
        """Validate settings and copy device names into an immutable tuple."""
        if isinstance(self.devices, (str, bytes)) or not isinstance(
            self.devices, Sequence
        ):
            raise TypeError("devices must be a sequence of device strings")
        if not self.devices:
            raise ValueError("devices must contain at least one device")
        if any(not isinstance(device, str) for device in self.devices):
            raise TypeError("each device must be a string")
        if any(
            not device.strip() or device != device.strip() for device in self.devices
        ):
            raise ValueError(
                "device names must be nonempty and have no outer whitespace"
            )
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        object.__setattr__(self, "devices", tuple(self.devices))


@dataclass(frozen=True)
class MultiProcessCrossEncoderConfig(MultiProcessEncoderConfig):
    """Execution settings for cross-encoder prediction with one replica per device.

    Uses the same device validation and immutable settings as
    :class:`MultiProcessEncoderConfig`. Each input counted by ``chunk_size`` is
    one query-document pair. Prediction ``batch_size`` controls the number of
    pairs processed in one inference batch within each worker.

    These are execution settings, not model experiment kwargs or result cache
    identities. Constructing this configuration does not load a model, check
    device availability, or start workers.

    Args:
        devices: Explicit worker devices, for example ``["cuda:0", "cuda:1"]``.
            Repeated ``"cpu"`` entries can be used for CPU workers.
        chunk_size: Number of query-document pairs dispatched as one worker job.
            Defaults to 2048.

    Examples:
        >>> config = MultiProcessCrossEncoderConfig(devices=["cuda:0", "cuda:1"])
        >>> config.devices
        ('cuda:0', 'cuda:1')
    """


def _collate_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {key: [row[key] for row in rows] for key in rows[0]}


def _worker_inputs(
    rows: list[dict[str, Any]], batch_size: int
) -> DataLoader[BatchedInput]:
    from datasets import Dataset
    from torch.utils.data import DataLoader

    return DataLoader(
        Dataset.from_list(rows),
        batch_size=batch_size,
        collate_fn=_collate_rows,
        num_workers=0,
    )


def _cpu_array(value: object) -> np.ndarray:
    """Copy dense results to CPU before sending them over a process boundary."""
    import torch

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(
                "Multiprocess execution currently supports dense outputs only"
            )
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    if not isinstance(value, np.ndarray) or value.dtype == object:
        raise TypeError("Multiprocess execution requires a dense numpy array or tensor")
    return value


def _model_worker(
    connection: Connection,
    meta: ModelMeta,
    device: str,
    model_kwargs: dict[str, Any],
    method: str,
) -> None:
    """Load once and service one job at a time on a dedicated device."""
    try:
        model = meta.load_model(device=device, **model_kwargs)
        if not callable(getattr(model, method, None)):
            raise TypeError(f"Model does not implement {method}()")
        connection.send(("ready", None))
        while True:
            job = connection.recv()
            if job is None:
                break
            operation, payload, kwargs = job
            if operation in {"encode", "predict"}:
                batch_size = kwargs.get("batch_size", 32)
                loaders = [_worker_inputs(rows, batch_size) for rows in payload]
                result = getattr(model, operation)(*loaders, **kwargs)
                if len(result) != len(payload[0]):
                    raise ValueError(
                        "Model returned a different number of outputs than inputs"
                    )
            else:
                result = getattr(model, operation)(*payload, **kwargs)
            connection.send(("result", _cpu_array(result)))
    except EOFError:
        pass
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _text_rows(inputs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Consume the actual loader order, preserving accompanying text fields."""
    for batch in inputs:
        if any(key in batch for key in ("image", "audio", "video")):
            raise ValueError(
                "Multiprocess execution currently supports text inputs only"
            )
        if "text" not in batch or any(
            not isinstance(text, str) for text in batch["text"]
        ):
            raise ValueError("Multiprocess execution requires batches of text strings")
        size = len(batch["text"])
        if any(len(values) != size for values in batch.values()):
            raise ValueError("Input batch columns have different lengths")
        for index in range(size):
            yield {key: values[index] for key, values in batch.items()}


def _chunks(
    inputs: Iterable[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    rows = _text_rows(inputs)
    while chunk := list(islice(rows, size)):
        yield chunk


class _MultiProcessModel:
    """Shared persistent worker lifecycle; public wrappers expose distinct protocols."""

    _method: str
    _config_type = MultiProcessEncoderConfig

    def __init__(
        self,
        model_meta: ModelMeta,
        config: MultiProcessEncoderConfig,
        *,
        timeout: float = 300,
        **model_kwargs: Any,
    ) -> None:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds")
        if "device" in model_kwargs or "device_map" in model_kwargs:
            raise ValueError(
                "Use config.devices for placement; device_map is unsupported"
            )
        resolved_meta = model_meta._prepare_load_model(**model_kwargs)
        if resolved_meta.loader_kwargs.get("device_map") is not None:
            raise ValueError(
                "Models using device_map cannot be replicated across workers"
            )
        if model_meta.modalities != ["text"]:
            raise ValueError(
                "Multiprocess execution currently supports text-only models"
            )
        if self._method == "encode" and any(
            kind in model_meta.model_type
            for kind in ("sparse", "late-interaction", "cross-encoder")
        ):
            raise ValueError("MultiProcessEncoder requires a dense encoder")
        self.mteb_model_meta = resolved_meta
        self.config = config
        self.timeout = timeout
        self._connections: list[Connection] = []
        self._processes: list[Any] = []
        self._closed = False

    @classmethod
    def from_model_meta(
        cls,
        model_meta: ModelMeta,
        *,
        devices: Sequence[str],
        chunk_size: int = 2048,
        timeout: float = 300,
        **model_kwargs: Any,
    ) -> Self:
        """Configure workers without loading a parent-process model."""
        return cls(
            model_meta,
            cls._config_type(devices, chunk_size),
            timeout=timeout,
            **model_kwargs,
        )

    def start(self) -> Self:
        """Start workers once; model loading happens in the spawned processes."""
        if self._closed:
            raise RuntimeError("Multiprocess model is closed")
        if self._processes:
            return self
        context = mp.get_context("spawn")
        try:
            for device in self.config.devices:
                parent, child = context.Pipe()
                self._connections.append(parent)
                process = context.Process(
                    target=_model_worker,
                    args=(child, self.mteb_model_meta, device, {}, self._method),
                )
                try:
                    process.start()
                finally:
                    child.close()
                self._processes.append(process)
            pending = dict.fromkeys(self._connections, time.monotonic())
            while pending:
                for connection in self._ready(pending):
                    status, _ = self._receive(connection)
                    if status != "ready":
                        raise RuntimeError("Worker did not acknowledge startup")
                    del pending[connection]
        except BaseException:
            self.close()
            raise
        return self

    def _ready(self, pending: dict[Connection, float]) -> list[Connection]:
        while True:
            if any(
                time.monotonic() - started > self.timeout
                for started in pending.values()
            ):
                devices = [
                    self.config.devices[self._connections.index(conn)]
                    for conn in pending
                ]
                raise TimeoutError(
                    f"Worker on {devices} did not respond within {self.timeout} seconds"
                )
            ready = wait(list(pending), timeout=0.1)
            if ready:
                return cast("list[Connection]", ready)
            for index, process in enumerate(self._processes):
                if not process.is_alive():
                    raise RuntimeError(
                        f"Worker on {self.config.devices[index]} exited with code {process.exitcode}"
                    )

    def _receive(self, connection: Connection) -> tuple[str, Any]:
        index = self._connections.index(connection)
        device = self.config.devices[index]
        try:
            status, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Worker on {device} disconnected") from error
        if status == "error":
            raise RuntimeError(f"Worker on {device} failed:\n{result}")
        return status, result

    def _run(
        self, jobs: Iterable[tuple[Any, ...]], kwargs: dict[str, Any]
    ) -> np.ndarray:
        from tqdm.auto import tqdm

        batch_size = kwargs.get("batch_size", 32)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        self.start()
        iterator = enumerate(jobs)
        pending: dict[Connection, float] = {}
        indices: dict[Connection, int] = {}
        results: dict[int, np.ndarray] = {}
        progress = tqdm(
            desc="Encoding" if self._method == "encode" else "Scoring pairs",
            unit="inputs",
            disable=not kwargs.get("show_progress_bar", True),
        )

        def submit(connection: Connection) -> None:
            job = next(iterator, None)
            if job is not None:
                index, payload = job
                connection.send(
                    (self._method, payload, {**kwargs, "show_progress_bar": False})
                )
                pending[connection] = time.monotonic()
                indices[connection] = index

        try:
            for connection in self._connections:
                submit(connection)
            while pending:
                for connection in self._ready(pending):
                    try:
                        _, result = self._receive(connection)
                    except RuntimeError as error:
                        raise RuntimeError(
                            f"Chunk {indices[connection]}: {error}"
                        ) from error
                    results[indices.pop(connection)] = result
                    progress.update(len(result))
                    del pending[connection]
                    submit(connection)
        except BaseException:
            self.close()
            raise
        finally:
            progress.close()
        if not results:
            dim = self.mteb_model_meta.embed_dim
            shape = (
                (0, dim if isinstance(dim, int) else 0)
                if self._method == "encode"
                else (0,)
            )
            return np.empty(shape, dtype=np.float32)
        return np.concatenate([results[index] for index in range(len(results))], axis=0)

    def close(self) -> None:
        """Stop and join workers, including after partial startup or failed jobs."""
        self._closed = True
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            process.close()
        for connection in self._connections:
            connection.close()
        self._processes.clear()
        self._connections.clear()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()


class MultiProcessEncoder(_MultiProcessModel):
    """Persistent dense text encoder workers, one model replica per device.

    Use as a context manager, or call ``close()`` when finished. Spawn requires
    an importable model loader and a script guarded by ``if __name__ == "__main__"``.
    Calls on a wrapper must be sequential. Each worker has at most one chunk in
    flight. Outputs are dense CPU numpy arrays in input order.
    """

    _method = "encode"

    def encode(self, inputs: DataLoader[BatchedInput], **kwargs: Any) -> Array:
        """Encode text chunks, preserving task metadata and prompt arguments."""
        return self._run(
            ((chunk,) for chunk in _chunks(inputs, self.config.chunk_size)), kwargs
        )

    def _similarity(self, method: str, first: Array, second: Array) -> Array:
        self.start()
        connection = self._connections[0]
        try:
            connection.send((method, (_cpu_array(first), _cpu_array(second)), {}))
            self._ready({connection: time.monotonic()})
            return cast("Array", self._receive(connection)[1])
        except BaseException:
            self.close()
            raise

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Use the model's own similarity implementation on the first worker."""
        return self._similarity("similarity", embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Use the model's own pairwise similarity implementation."""
        return self._similarity("similarity_pairwise", embeddings1, embeddings2)


class MultiProcessCrossEncoder(_MultiProcessModel):
    """Persistent text cross-encoder workers returning scores in pair order.

    Lifecycle and spawn requirements are the same as :class:`MultiProcessEncoder`.
    Inputs are paired by row, even when their dataloader batch sizes differ.
    """

    _method = "predict"
    _config_type = MultiProcessCrossEncoderConfig

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        **kwargs: Any,
    ) -> Array:
        """Score matching text pairs; reject unequal input counts."""
        jobs = zip(
            _chunks(inputs1, self.config.chunk_size),
            _chunks(inputs2, self.config.chunk_size),
            strict=True,
        )
        return self._run(jobs, kwargs)
