from __future__ import annotations

import torch

from mteb.types import Array


def use_torch_compile():
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    return gpu_ok


def convert_to_tensor(a: Array, dtype=torch.float32) -> torch.Tensor:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=dtype)
    return a


def normalize_embeddings(embeddings: Array) -> torch.Tensor:
    """Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings: The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    embeddings = convert_to_tensor(embeddings)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def cos_sim(a: Array, b: Array) -> torch.Tensor:
    """Calculate pairwise cosine similarities between two sets of vectors.

    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    # since compile works better with pure tensor operations
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)

    # The actual function to compile
    def _cos_sim_core(a_tensor, b_tensor):
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.reshape(1, *a_tensor.shape)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.reshape(1, *b_tensor.shape)

        a_norm = normalize_embeddings(a_tensor)
        b_norm = normalize_embeddings(b_tensor)
        return a_norm @ b_norm.transpose(0, 1)

    # Compile the core function once
    should_compile = (
        hasattr(torch, "compile")
        and use_torch_compile()
        and (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor))
    )
    if should_compile:
        _cos_sim_core_compiled = torch.compile(_cos_sim_core)
        return _cos_sim_core_compiled(a, b)
    else:
        return _cos_sim_core(a, b)


# https://github.com/UKPLab/sentence-transformers/blob/3fd59c3d122f2148e22b6338447b45d850fb6ea4/sentence_transformers/util.py#L125
def pairwise_cos_sim(a: Array, b: Array) -> Array:
    """Computes the pairwise cosine similarity cos_sim(a[i], b[i]).

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Tensor: Vector with res[i] = cos_sim(a[i], b[i])
    """
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def max_sim(a: Array, b: Array) -> torch.Tensor:
    """Computes the max-similarity max_sim(a[i], b[j]) for all i and j.
    Works with a Tensor of the shape (batch_size, num_tokens, token_dim)

    Return:
        Matrix with res[i][j]  = max_sim(a[i], b[j])
    """  # noqa: D402
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)

    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)  # eq. to a.unsqueeze(0)

    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)

    scores = torch.einsum(
        "ash,bth->abst",
        a,
        b,
    )

    return scores.max(axis=-1).values.sum(axis=-1)


# https://github.com/lightonai/pylate/blob/2d094a724866d6e15701781528368438081c0157/pylate/scores/scores.py#L67C1-L122C38
def pairwise_max_sim(
    queries_embeddings: Array,
    documents_embeddings: Array,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities
    between the query and the document for corresponding pairs.

    Args:
        queries_embeddings: The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
        documents_embeddings: The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)
    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = convert_to_tensor(query_embedding)
        document_embedding = convert_to_tensor(document_embedding)

        query_document_score = torch.einsum(
            "sh,th->st",
            query_embedding,
            document_embedding,
        )

        scores.append(query_document_score.max(axis=-1).values.sum())

    return torch.stack(scores, dim=0)


def dot_score(a: Array, b: Array) -> torch.Tensor:
    """Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    # Move tensor conversion outside the compiled function
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)

    # The actual function to compile
    def _dot_score_core(a_tensor, b_tensor):
        if len(a_tensor.shape) == 1:
            a_tensor = a_tensor.unsqueeze(0)
        if len(b_tensor.shape) == 1:
            b_tensor = b_tensor.unsqueeze(0)

        return a_tensor @ b_tensor.transpose(0, 1)

    # Compile the core function once
    if (
        hasattr(torch, "compile")
        and use_torch_compile()
        and isinstance(a, torch.Tensor)
    ):
        _dot_score_core_compiled = torch.compile(_dot_score_core)
        return _dot_score_core_compiled(a, b)
    else:
        return _dot_score_core(a, b)


def pairwise_dot_score(a: Array, b: Array) -> Array:
    """Computes the pairwise dot-product dot_prod(a[i], b[i]).

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = dot_prod(a[i], b[i])
    """
    return (a * b).sum(dim=-1)


# https://github.com/UKPLab/sentence-transformers/blob/3fd59c3d122f2148e22b6338447b45d850fb6ea4/sentence_transformers/util.py#L196C1-L227C56
def euclidean_sim(a: Array, b: Array) -> Array:
    """Computes the euclidean similarity (i.e., negative distance) between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = -euclidean_distance(a[i], b[j])
    """
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)

    return -torch.cdist(a, b, p=2.0)


def pairwise_euclidean_sim(a: Array, b: Array) -> Array:
    """Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)

    return -torch.sqrt(torch.sum((a - b) ** 2, dim=-1))


def vision_similarity(text_embeddings: Array, image_embeddings: Array) -> Array:
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    logits = torch.matmul(image_embeddings, text_embeddings.T)
    probs = (logits * 100).softmax(dim=-1)
    return probs
