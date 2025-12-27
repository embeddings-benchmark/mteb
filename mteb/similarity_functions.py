import torch

from mteb.models import EncoderProtocol
from mteb.models.model_meta import ScoringFunction
from mteb.types import Array


def _use_torch_compile():
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True

    return gpu_ok


def _convert_to_tensor(a: Array, dtype=torch.float32) -> torch.Tensor:
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=dtype)
    return a


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
    elif similarity_fn is ScoringFunction.DOT_PRODUCT:
        return dot_score(embedding1, embedding2)
    elif similarity_fn is ScoringFunction.EUCLIDEAN:
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
    elif similarity_fn is ScoringFunction.DOT_PRODUCT:
        return pairwise_dot_score(embedding1, embedding2)
    elif similarity_fn is ScoringFunction.EUCLIDEAN:
        return pairwise_euclidean_sim(embedding1, embedding2)
    raise ValueError(f"Unsupported similarity function: {similarity_fn}")


def _normalize_embeddings(embeddings: Array) -> torch.Tensor:
    """Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings: The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
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
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    # The actual function to compile
    def _cos_sim_core(a_tensor, b_tensor):
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
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    return pairwise_dot_score(_normalize_embeddings(a), _normalize_embeddings(b))


def max_sim(a: Array, b: Array) -> torch.Tensor:
    """Compute the maximum pairwise similarity between tokens.

    Given two tensors `a` and `b` of shape (batch_size, num_tokens, token_dim),
    this function computes the maximum similarity `max_sim(a[i], b[j])` for all
    pairs of tokens `i` and `j` across the two inputs.

    Args:
        a: Tensor of shape (batch_size, num_tokens, token_dim).
        b: Tensor of shape (batch_size, num_tokens, token_dim).

    Returns:
        A tensor containing the maximum similarity values for each batch.
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)  # eq. to a.unsqueeze(0)

    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)

    scores = torch.einsum(
        "ash,bth->abst",
        a,
        b,
    )

    return scores.max(axis=-1).values.sum(axis=-1)  # type: ignore[call-overload]


# https://github.com/lightonai/pylate/blob/2d094a724866d6e15701781528368438081c0157/pylate/scores/scores.py#L67C1-L122C38
def pairwise_max_sim(
    queries_embeddings: Array,
    documents_embeddings: Array,
) -> torch.Tensor:
    """Computes the ColBERT score for each query-document pair. The score is computed as the sum of maximum similarities between the query and the document for corresponding pairs.

    Args:
        queries_embeddings: The first tensor. The queries embeddings. Shape: (batch_size, num tokens queries, embedding_size)
        documents_embeddings: The second tensor. The documents embeddings. Shape: (batch_size, num tokens documents, embedding_size)

    Returns:
        Tensor: Vector with res[i] = max_sim(queries_embeddings[i], documents_embeddings[i])
    """
    scores = []

    for query_embedding, document_embedding in zip(
        queries_embeddings, documents_embeddings
    ):
        query_embedding = _convert_to_tensor(query_embedding)
        document_embedding = _convert_to_tensor(document_embedding)

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
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

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
        and _use_torch_compile()
        and isinstance(a, torch.Tensor)
    ):
        _dot_score_core_compiled = torch.compile(_dot_score_core)
        return _dot_score_core_compiled(a, b)
    else:
        return _dot_score_core(a, b)


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
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    return -torch.cdist(a, b, p=2.0)


def pairwise_euclidean_sim(a: Array, b: Array) -> Array:
    """Computes the euclidean distance (i.e., negative distance) between pairs of tensors.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        Vector with res[i] = -euclidean_distance(a[i], b[i])
    """
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
    text_embeddings_tensor = _convert_to_tensor(text_embeddings)
    input_embeddings_tensor = _convert_to_tensor(input_embeddings)

    text_embeddings_tensor = text_embeddings_tensor / text_embeddings_tensor.norm(
        dim=-1, keepdim=True
    )
    input_embeddings_tensor = input_embeddings_tensor / input_embeddings_tensor.norm(
        dim=-1, keepdim=True
    )
    logits = torch.matmul(input_embeddings_tensor, text_embeddings_tensor.T)
    probs = (logits * 100).softmax(dim=-1)
    return probs
