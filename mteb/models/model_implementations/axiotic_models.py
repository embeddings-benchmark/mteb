from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


class OgmaWrapper:
    """MTEB wrapper for Axiotic Ogma embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        **_: Any,
    ) -> None:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = Path(snapshot_download(model_name, revision=revision))

        # The Ogma HF repos ship their lightweight model implementation as source files.
        import sys

        sys.path.insert(0, str(self.model_path))
        from config import TaskToken  # type: ignore[import-not-found]
        from ogma_model import OgmaModel  # type: ignore[import-not-found]

        self.task_token = TaskToken
        self.model = OgmaModel.from_checkpoint(
            str(self.model_path), device=str(self.device)
        )
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        self.n_special_tokens = 7
        self.max_length = int(getattr(self.model.config, "max_seq_len", 1024))

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def _task_for_prompt(self, prompt_type: PromptType | None) -> Any:
        if prompt_type == PromptType.query:
            return self.task_token.QRY
        if prompt_type == PromptType.document:
            return self.task_token.DOC
        return self.task_token.SYM

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: list[list[int]] = []
        for text in texts:
            enc = self.tokenizer.encode(text)
            ids = enc.ids
            tokens = enc.tokens
            if tokens and tokens[0] in {"[CLS]", "<s>"}:
                ids = ids[1:]
                tokens = tokens[1:]
            if tokens and tokens[-1] in {"[SEP]", "</s>"}:
                ids = ids[:-1]
            ogma_ids = (
                [2] + [token_id + self.n_special_tokens for token_id in ids] + [3]
            )
            encoded.append(ogma_ids[: self.max_length])

        max_len = max(len(ids) for ids in encoded)
        token_ids = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            len(encoded), max_len, dtype=torch.long, device=self.device
        )
        for row, ids in enumerate(encoded):
            tensor_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            token_ids[row, : len(ids)] = tensor_ids
            attention_mask[row, : len(ids)] = 1
        return token_ids, attention_mask

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **_: Any,
    ) -> Array:
        del task_metadata, hf_split, hf_subset
        task = self._task_for_prompt(prompt_type)
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for batch in inputs:
                texts = [str(text) for text in batch["text"]]
                token_ids, attention_mask = self._tokenize(texts)
                batch_embeddings = self.model.encode(
                    token_ids, attention_mask, task=task
                )
                embeddings.append(batch_embeddings.detach().cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)


ogma_micro = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-micro",
    model_type=["dense"],
    revision="c6ec1216f5ca032ad1916b07eb25529649dd99f1",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=2_323_200,
    memory_usage_mb=8.9,
    max_tokens=1024,
    embed_dim=128,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-micro",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_mini = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-mini",
    model_type=["dense"],
    revision="668d483b334d164a9aaaab3404a1b7f4a60a8329",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=3_511_808,
    memory_usage_mb=13.4,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-mini",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_small = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-small",
    model_type=["dense"],
    revision="e33b8f6088d74f5accda110ac92688079d38ae48",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=8_596_544,
    memory_usage_mb=32.8,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-small",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_base = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-base",
    model_type=["dense"],
    revision="66c255277fb755812583c4abadc69fcd7504cb85",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=13_318_400,
    memory_usage_mb=50.8,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)

ogma_large = ModelMeta(
    loader=OgmaWrapper,
    name="axiotic/ogma-large",
    model_type=["dense"],
    revision="f54d08762f2eb2ba1fec9a12e390d02e759e3a7f",
    release_date="2026-04-23",
    languages=["eng-Latn"],
    open_weights=True,
    framework=["PyTorch", "safetensors"],
    n_parameters=32_365_472,
    memory_usage_mb=123.5,
    max_tokens=1024,
    embed_dim=256,
    license="cc-by-nc-4.0",
    reference="https://huggingface.co/axiotic/ogma-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
