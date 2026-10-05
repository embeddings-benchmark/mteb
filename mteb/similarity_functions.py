from __future__ import annotations

import logging
from math import isqrt
from typing import TYPE_CHECKING, Any, cast

from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch

    from mteb.models import EncoderProtocol
    from mteb.types import Array

logger = logging.getLogger(__name__)

# Element budget for max_sim's (queries, documents, query_tokens, document_tokens) intermediate,
# about 400 MB in float32.
_MAX_SIM_CHUNK_ELEMENTS = 100_000_000

# Padded tokens per chunk of queries or documents, so that a chunk pair stays within the element budget.
_MAX_SIM_CHUNK_TOKENS = isqrt(_MAX_SIM_CHUNK_ELEMENTS)

# Ragged inputs are grouped by length only if padding them all would compare more than this many
# times the number of real token pairs.
_MAX_SIM_PADDING_OVERHEAD = 2


def _use_torch_compile() -> bool:
    import torch

    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):  # noqa: PLR6201
            gpu_ok = True

    return gpu_ok


def _convert_to_tensor(a: Array, dtype: torch.dtype | None = None) -> torch.Tensor:
    import torch

    dtype = torch.float32 if dtype is None else dtype

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=dtype)
    elif torch.is_floating_point(a) and torch.finfo(a.dtype).bits < 32:
        a = a.to(
            torch.float32
        )  # upcast sub-float32 floats (fp8/float16/bfloat16) to break ties
    return a


def _select_device(a: torch.Tensor, b: torch.Tensor) -> torch.device:
    """Pick the device to score `a` against `b` on.

    Inputs already on an accelerator are scored there. CPU inputs are moved to CUDA or MPS
    when one is available and the amount of work (multiply-adds) is large enough to pay for the copy.
    """
    import torch

    if a.device.type != "cpu":
        return a.device
    if b.device.type != "cpu":
        return b.device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.float64 not in {a.dtype, b.dtype}:
        return torch.device("mps")
    return a.device


def _compute_on_best_device(
    core_fn: Callable[[Any, Any, torch.device], torch.Tensor],
    a: torch.Tensor | list[torch.Tensor],
    b: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:
    """Run `core_fn(a, b, device)` on the selected device and return the result on `a`'s device.

    `a` and `b` are tensors or non-empty lists of tensors. `core_fn` moves its inputs to `device`
    itself, so it can stream large inputs in chunks. Falls back to the CPU if the accelerator runs
    out of memory.
    """
    import torch

    def first(x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        return x[0] if isinstance(x, list) else x

    def to(
        x: torch.Tensor | list[torch.Tensor], device: torch.device
    ) -> torch.Tensor | list[torch.Tensor]:
        return [t.to(device) for t in x] if isinstance(x, list) else x.to(device)

    a_device = first(a).device
    b_device = first(b).device
    device = _select_device(first(a), first(b))
    if device == a_device == b_device:
        return core_fn(a, b, device)
    try:
        return core_fn(a, b, device).to(a_device)
    except RuntimeError as e:
        # torch.OutOfMemoryError (CUDA) subclasses RuntimeError; MPS raises a plain RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        logger.warning(
            f"Ran out of memory computing similarity on {device}, falling back to CPU."
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        cpu = torch.device("cpu")
        return core_fn(to(a, cpu), to(b, cpu), cpu).to(a_device)


def compute_pairwise_similarity(
    model: EncoderProtocol, embedding1: Array, embedding2: Array
) -> Array:
    """Compute pairwise similarity between two sets of embeddings using the model's built-in similarity function if available, otherwise using cosine similarity.

    Args:
        model: An instance of EncoderProtocol which may have a custom similarity function.
        embedding1: The first set of embeddings.
        embedding2: The second set of embeddings.

    Returns:
        Array: The computed pairwise similarity scores.
    """
    if hasattr(model, "similarity_pairwise"):
        return model.similarity_pairwise(embedding1, embedding2)
    return pairwise_cos_sim(embedding1, embedding2)


def select_similarity(
    embedding1: Array,
    embedding2: Array,
    similarity_fn: ScoringFunction,
) -> Array:
    """Compute similarity between two sets of embeddings using the specified similarity function.

    Args:
        embedding1: The first set of embeddings.
        embedding2: The second set of embeddings.
        similarity_fn: The similarity function to use (COSINE, DOT_PRODUCT, EUCLIDEAN).

    Returns:
        Array: The computed similarity scores.
    """
    if similarity_fn is ScoringFunction.COSINE:
        return cos_sim(embedding1, embedding2)
    if similarity_fn is ScoringFunction.DOT_PRODUCT:
        return dot_score(embedding1, embedding2)
    if similarity_fn is ScoringFunction.EUCLIDEAN:
        return euclidean_sim(embedding1, embedding2)
    raise ValueError(f"Unsupported similarity function: {similarity_fn}")


def select_pairwise_similarity(
    embedding1: Array,
    embedding2: Array,
    similarity_fn: ScoringFunction,
) -> Array:
    """Compute pairwise similarity between two sets of embeddings using the specified similarity function.

    Args:
        embedding1: The first set of embeddings.
        embedding2: The second set of embeddings.
        similarity_fn: The similarity function to use (COSINE, DOT_PRODUCT, EUCLIDEAN).

    Returns:
        Array: The computed pairwise similarity scores.
    """
    if similarity_fn is ScoringFunction.COSINE:
        return pairwise_cos_sim(embedding1, embedding2)
    if similarity_fn is ScoringFunction.DOT_PRODUCT:
        return pairwise_dot_score(embedding1, embedding2)
    if similarity_fn is ScoringFunction.EUCLIDEAN:
        return pairwise_euclidean_sim(embedding1, embedding2)
    raise ValueError(f"Unsupported similarity function: {similarity_fn}")


def _normalize_embeddings(embeddings: Array) -> torch.Tensor:
    """Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings: The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    import torch

    embeddings = _convert_to_tensor(embeddings)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def cos_sim(a: Array, b: Array) -> torch.Tensor:
    """Calculate pairwise cosine similarities between two sets of vectors.

    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    # since compile works better with pure tensor operations
    import torch

    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    # The actual function to compile
    def _cos_sim_core(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.reshape(1, *a_tensor.shape)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.reshape(1, *b_tensor.shape)

        a_norm = _normalize_embeddings(a_tensor)
        b_norm = _normalize_embeddings(b_tensor)
        return a_norm @ b_norm.transpose(0, 1)

    # Compile the core function once
    should_compile = (
        hasattr(torch, "compile")
        and _use_torch_compile()
        and (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor))
    )
    core_fn = torch.compile(_cos_sim_core) if should_compile else _cos_sim_core
    return _compute_on_best_device(
        lambda a_, b_, device: core_fn(a_.to(device), b_.to(device)), a, b
    )


# https://github.com/UKPLab/sentence-transformers/blob/3fd59c3d122f2148e22b6338447b45d850fb6ea4/sentence_transformers/util.py#L125
def pairwise_cos_sim(a: Array, b: Array) -> Array:
    """Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    return pairwise_dot_score(_normalize_embeddings(a), _normalize_embeddings(b))


def _is_ragged(x: Array | Sequence[Array]) -> bool:
    """Whether `x` is a non-empty list of 2D `(num_tokens, token_dim)` embeddings."""
    return (
        isinstance(x, (list, tuple)) and len(x) > 0 and getattr(x[0], "ndim", None) == 2
    )


def _as_token_tensors(x: Array | Sequence[Array]) -> list[torch.Tensor]:
    """Split `x` into one 2D `(num_tokens, token_dim)` tensor per input."""
    import torch

    if _is_ragged(x):
        return [_convert_to_tensor(t) for t in cast("Sequence[Array]", x)]
    tensor = _convert_to_tensor(cast("Array", x))
    return [tensor] if tensor.ndim == 2 else list(torch.unbind(tensor))


def _token_budget_chunks(
    widths: Sequence[int], max_tokens: int = _MAX_SIM_CHUNK_TOKENS
) -> list[list[int]]:
    """Group input indices by length so each group, padded to its widest input, fits `max_tokens`.

    Indices are sorted by width and added to a group until `group size * widest width` would exceed
    `max_tokens`. An input wider than `max_tokens` gets its own group.
    """
    order = sorted(range(len(widths)), key=widths.__getitem__)
    chunks: list[list[int]] = []
    current: list[int] = []
    for index in order:
        # Ascending order, so the newest input is the widest in the group
        if current and (len(current) + 1) * widths[index] > max_tokens:
            chunks.append(current)
            current = []
        current.append(index)
    if current:
        chunks.append(current)
    return chunks


def _max_sim_ragged_core(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    device: torch.device,
    max_chunk_tokens: int = _MAX_SIM_CHUNK_TOKENS,
) -> torch.Tensor:
    """Compute MaxSim between two lists of `(num_tokens, token_dim)` tensors of varying length.

    Queries and documents are grouped by length (see `_token_budget_chunks`) and zero-padded within
    their group. Query groups are moved to `device` once, and each document group is scored against
    all of them in turn. Returns a `(len(a), len(b))` tensor on `device`.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    out = torch.empty(
        len(a),
        len(b),
        dtype=torch.promote_types(a[0].dtype, b[0].dtype),
        device=device,
    )
    query_chunks = [
        (
            torch.as_tensor(indices, device=device),
            pad_sequence([a[i].to(device) for i in indices], batch_first=True),
        )
        for indices in _token_budget_chunks([len(t) for t in a], max_chunk_tokens)
    ]
    for doc_indices in _token_budget_chunks([len(t) for t in b], max_chunk_tokens):
        doc_index = torch.as_tensor(doc_indices, device=device)
        documents = pad_sequence(
            [b[i].to(device) for i in doc_indices], batch_first=True
        )
        for query_index, queries in query_chunks:
            scores = torch.einsum("ash,bth->abst", queries, documents)
            maxima = scores.max(axis=-1).values.sum(axis=-1)  # type: ignore[call-overload]
            out[query_index[:, None], doc_index[None, :]] = maxima
    return out


def max_sim(
    a: Array | Sequence[Array], b: Array | Sequence[Array], batch_size: int = 128
) -> torch.Tensor:
    """Compute the maximum pairwise similarity between tokens.

    Given two tensors `a` and `b` of shape (batch_size, num_tokens, token_dim),
    this function computes the maximum similarity `max_sim(a[i], b[j])` for all
    pairs of tokens `i` and `j` across the two inputs.

    Either input may also be a list of `(num_tokens, token_dim)` tensors with varying `num_tokens`.
    These are zero-padded, and if that would compare more than `_MAX_SIM_PADDING_OVERHEAD` times the
    real token pairs (e.g. one long query among short ones), they are padded per group of similar
    length instead. Both give the same scores.

    `a` is scored in `batch_size`-sized chunks and `b` in chunks sized so that the
    `(a_chunk, b_chunk, num_tokens_a, num_tokens_b)` intermediate from each `einsum` call
    stays within `_MAX_SIM_CHUNK_ELEMENTS`, instead of holding the full intermediate in memory.

    Args:
        a: Tensor of shape (batch_size, num_tokens, token_dim), or a list of
            `(num_tokens, token_dim)` tensors.
        b: Tensor of shape (batch_size, num_tokens, token_dim), or a list of
            `(num_tokens, token_dim)` tensors.
        batch_size: Maximum number of rows of `a` to score at a time. Not used when ragged inputs
            are grouped by length.

    Returns:
        A tensor containing the maximum similarity values for each batch.
    """
    import torch

    if _is_ragged(a) or _is_ragged(b):
        a_tokens = _as_token_tensors(a)
        b_tokens = _as_token_tensors(b)
        a_widths = [len(t) for t in a_tokens]
        b_widths = [len(t) for t in b_tokens]
        # token pairs compared when padding everything to the longest input, vs. real ones
        padded = len(a_widths) * max(a_widths) * len(b_widths) * max(b_widths)
        real = sum(a_widths) * sum(b_widths)
        if padded > _MAX_SIM_PADDING_OVERHEAD * real:
            return _compute_on_best_device(_max_sim_ragged_core, a_tokens, b_tokens)
        from torch.nn.utils.rnn import pad_sequence

        a = pad_sequence(a_tokens, batch_first=True)
        b = pad_sequence(b_tokens, batch_first=True)

    a = _convert_to_tensor(cast("Array", a))
    b = _convert_to_tensor(cast("Array", b))

    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)  # eq. to a.unsqueeze(0)

    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)

    def _max_sim_core(
        a_tensor: torch.Tensor, b_tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        out = torch.empty(
            a_tensor.size(0),
            b_tensor.size(0),
            dtype=torch.promote_types(a_tensor.dtype, b_tensor.dtype),
            device=a_tensor.device,
        )
        # Bound the rows of `a` too: with long tokens on both sides, even one row of `b`
        # against `batch_size` rows of `a` can exceed the budget.
        tokens_per_pair = max(1, a_tensor.size(1) * b_tensor.size(1))
        a_rows = max(
            1,
            min(
                batch_size,
                a_tensor.size(0),
                _MAX_SIM_CHUNK_ELEMENTS // tokens_per_pair,
            ),
        )
        b_chunk_size = max(1, _MAX_SIM_CHUNK_ELEMENTS // (a_rows * tokens_per_pair))
        for b_start in range(0, b_tensor.size(0), b_chunk_size):
            b_chunk = b_tensor[b_start : b_start + b_chunk_size].to(device)
            for a_start in range(0, a_tensor.size(0), a_rows):
                a_chunk = a_tensor[a_start : a_start + a_rows].to(device)
                scores = torch.einsum("ash,bth->abst", a_chunk, b_chunk)
                out[
                    a_start : a_start + a_chunk.size(0),
                    b_start : b_start + b_chunk.size(0),
                ] = scores.max(axis=-1).values.sum(axis=-1)  # type: ignore[call-overload]
        return out

    return _compute_on_best_device(_max_sim_core, a, b)


# https://github.com/lightonai/pylate/blob/2d094a724866d6e15701781528368438081c0157/pylate/scores/scores.py#L67C1-L122C38
def pairwise_max_sim(
    queries_embeddings: Array | Sequence[Array],
    documents_embeddings: Array | Sequence[Array],
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities between the query and the document for corresponding pairs.

    Args:
        queries_embeddings: The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
        documents_embeddings: The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Returns:
        Tensor: Vector with res[i] = max_sim(queries_embeddings[i], documents_embeddings[i])
    """
    import torch

    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings, strict=True
    ):
        query_embedding = _convert_to_tensor(query_embedding)  # noqa: PLW2901
        document_embedding = _convert_to_tensor(document_embedding)  # noqa: PLW2901

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())  # type: ignore[call-overload]

    return torch.stack(scores, dim=0)


def dot_score(a: Array, b: Array) -> torch.Tensor:
    """Calculate pairwise dot products between two sets of vectors.

    Computes the dot product dot_prod(a[i], b[j]) for all i and j.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    import torch

    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    # The actual function to compile
    def _dot_score_core(a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> torch.Tensor:
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.unsqueeze(0)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.unsqueeze(0)

        return a_tensor @ b_tensor.transpose(0, 1)

    # Compile the core function once
    should_compile = (
        hasattr(torch, "compile")
        and _use_torch_compile()
        and isinstance(a, torch.Tensor)
    )
    core_fn = torch.compile(_dot_score_core) if should_compile else _dot_score_core
    return _compute_on_best_device(
        lambda a_, b_, device: core_fn(a_.to(device), b_.to(device)), a, b
    )


def pairwise_dot_score(a: Array, b: Array) -> Array:
    """Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    return (a * b).sum(dim=-1)


# https://github.com/UKPLab/sentence-transformers/blob/3fd59c3d122f2148e22b6338447b45d850fb6ea4/sentence_transformers/util.py#L196C1-L227C56
def euclidean_sim(a: Array, b: Array) -> Array:
    """Computes the euclidean similarity (i.e., negative distance) between two tensors.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -euclidean_distance(a[i], b[j])
    """
    import torch

    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return _compute_on_best_device(
        lambda a_, b_, device: -torch.cdist(a_.to(device), b_.to(device), p=2.0), a, b
    )


def pairwise_euclidean_sim(a: Array, b: Array) -> Array:
    """Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
    import torch

    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def similarity(text_embeddings: Array, input_embeddings: Array) -> Array:
    """Similarity function used in ImageTextPair classification

    Args:
        text_embeddings: Embeddings of the text inputs
        input_embeddings: Embeddings of the image inputs

    Returns:
        Matrix with similarities
    """
    import torch

    text_embeddings_tensor = _convert_to_tensor(text_embeddings)
    input_embeddings_tensor = _convert_to_tensor(input_embeddings)

    text_embeddings_tensor = text_embeddings_tensor / text_embeddings_tensor.norm(  # noqa: PLR6104
        dim=-1, keepdim=True
    )
    input_embeddings_tensor = input_embeddings_tensor / input_embeddings_tensor.norm(  # noqa: PLR6104
        dim=-1, keepdim=True
    )
    logits = torch.matmul(input_embeddings_tensor, text_embeddings_tensor.T)
    probs = (logits * 100).softmax(dim=-1)
    return probs
