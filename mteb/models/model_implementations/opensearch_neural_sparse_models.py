from typing import Any

import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

v2_training_data = {
    "MSMARCO",
    # not in MTEB. see https://huggingface.co/datasets/sentence-transformers/embedding-training-data
    # "eli5_question_answer",
    # "gooaq_pairs",
    # "searchQA_top5_snippets",
    # "squad_pairs",
    # "stackexchange_duplicate_questions_body_body",
    # "stackexchange_duplicate_questions_title_title",
    # "stackexchange_duplicate_questions_title-body_title-body",
    # "WikiAnswers",
    # "wikihow",
    # "yahoo_answers_question_answer",
    # "yahoo_answers_title_answer",
    # "yahoo_answers_title_question",
}


v3_training_data = v2_training_data | {
    "HotpotQA",
    "FEVER",
    "FIQA",
    "NFCORPUS",
    "SCIFACT",
    # not in MTEB. see https://huggingface.co/datasets/sentence-transformers/embedding-training-data
    # "NQ-train_pairs",
    # "quora_duplicates",
}


class SparseEncoderWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        import sentence_transformers

        if sentence_transformers.__version__ < "5.0.0":
            raise ImportError(
                "sentence-transformers version must be >= 5.0.0 to load sparse encoder"
            )
        from sentence_transformers.sparse_encoder import SparseEncoder

        self.model_name = model_name
        self.kwargs = kwargs
        self.model = SparseEncoder(model_name, **kwargs)
        self.model.to(torch_dtype)
        self.batch_size = kwargs.get("batch_size", 1000)

    def similarity(
        self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between sparse query_embeddings and corpus_embeddings in batches.

        Args:
            query_embeddings: sparse COO tensor of shape (num_queries, dim)
            corpus_embeddings: tensor of shape (num_corpus, dim)

        Returns:
            Similarity matrix of shape (num_queries, num_corpus)
        """
        sims = []
        num_queries = query_embeddings.size(0)
        batch_size = self.batch_size

        # Ensure query_embeddings is coalesced sparse COO
        q = query_embeddings.coalesce()
        indices = q.indices()  # 2 x nnz: [row, col]
        values = q.values()  # nnz
        n_cols = q.size(1)

        # Iterate over sparse query embeddings in batches
        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            # Select non-zero entries for this batch
            mask = (indices[0] >= start) & (indices[0] < end)
            sel_idx = indices[:, mask].clone()
            sel_idx[0] -= start  # shift row indices to batch-local
            sel_vals = values[mask].clone()

            # Build sparse batch tensor of shape (batch_rows, dim)
            batch_q = torch.sparse_coo_tensor(
                sel_idx,
                sel_vals,
                size=(end - start, n_cols),
                device=q.device,
                dtype=q.dtype,
            ).coalesce()

            # Compute similarity for this sparse batch
            sim_batch = self.model.similarity(batch_q, corpus_embeddings)
            sims.append(sim_batch)

        # Concatenate all batch results
        return torch.cat(sims, dim=0)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        sentences = [text for batch in inputs for text in batch["text"]]

        if prompt_type is not None and prompt_type == PromptType.query:
            return self.model.encode_query(
                sentences,  # type: ignore[arg-type]
                **kwargs,
            )
        return self.model.encode_document(sentences, **kwargs)  # type: ignore[return-value]


opensearch_neural_sparse_encoding_doc_v3_gte = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    languages=["eng-Latn"],
    open_weights=True,
    revision="a8abaa916125ee512a7a8f4d706d07eb0128a8e6",
    release_date="2025-06-18",
    n_parameters=137_394_234,
    memory_usage_mb=549,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v3_training_data,
    loader=SparseEncoderWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
    ),
)


opensearch_neural_sparse_encoding_doc_v3_distill = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    languages=["eng-Latn"],
    open_weights=True,
    revision="babf71f3c48695e2e53a978208e8aba48335e3c0",
    release_date="2025-03-28",
    n_parameters=66_985_530,
    memory_usage_mb=267,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v3_training_data,
    loader=SparseEncoderWrapper,
)

opensearch_neural_sparse_encoding_doc_v2_distill = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    languages=["eng-Latn"],
    open_weights=True,
    revision="8921a26c78b8559d6604eb1f5c0b74c079bee38f",
    release_date="2024-07-17",
    n_parameters=66_985_530,
    memory_usage_mb=267,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v2_training_data,
    loader=SparseEncoderWrapper,
)


opensearch_neural_sparse_encoding_doc_v2_mini = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4af867a426867dfdd744097531046f4289a32fdd",
    release_date="2024-07-18",
    n_parameters=22_744_506,
    memory_usage_mb=86,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets=v2_training_data,
    loader=SparseEncoderWrapper,
)

opensearch_neural_sparse_encoding_doc_v1 = ModelMeta(
    name="opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="98cdcbd72867c547f72f2b7b7bed9cdf9f09922d",
    release_date="2024-03-07",
    n_parameters=132_955_194,
    memory_usage_mb=507,
    embed_dim=30522,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
    similarity_fn_name="dot",
    framework=["Sentence Transformers", "PyTorch"],
    public_training_code="https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample",
    public_training_data=True,
    use_instructions=True,
    training_datasets={
        "MSMARCO",
    },
    loader=SparseEncoderWrapper,
)
