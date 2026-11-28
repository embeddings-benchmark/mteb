"""hyper3-embed-deu by hyper3labs: German text embeddings fine-tuned from Gemma 4 E2B.

Loads the Hub repository's ``Hyper3EmbedModel`` via ``AutoModel.from_pretrained``
(``trust_remote_code=True``), restoring the bidirectional sliding window, appending EOS, and
pooling with an FP32 latent cross-attention block (excluding prompt tokens from the mean).

By default (``multivector=False``), ``Hyper3EmbedWrapper`` encodes 1,536-d normalized FP32
embeddings (up to 4,096 tokens) scored with cosine similarity. Passing ``multivector=True`` enables
the model's hybrid dense + 128-d MaxSim retrieval scorer over full documents and overlapping
512-token windows (stride 256).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.models.abs_encoder import AbsEncoder, get_prompt
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import dot_score, max_sim
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

# Settings of the model's hybrid retrieval scorer (modeling_hyper3_embed.py in the Hub repository).
_QUERY_MAX_LENGTH = 1024
_WINDOW_MAX_LENGTH = 512 + 64
_WINDOW_BLOCK = 4096
_DENSE_WEIGHT = 0.25

HYPER3_EMBED_DEU_PROMPTS: dict[str, str] = {
    # Task types (mteb prompt-key order: task-role, task, type-role, type, role)
    "Retrieval-query": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
    "Retrieval-document": "",
    "Retrieval": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
    "Reranking-query": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
    "Reranking-document": "",
    "Reranking": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
    "Classification": "Instruct: Classify the text.\nQuery: ",
    "Classification-document": "",
    "MultilabelClassification": "Instruct: Classify the text.\nQuery: ",
    "MultilabelClassification-document": "",
    "Clustering": "Instruct: Group texts by topic.\nQuery: ",
    "STS": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "PairClassification": "Instruct: Determine whether texts have the same meaning.\nQuery: ",
    "BitextMining": "Instruct: Retrieve translation-equivalent text.\nQuery: ",
    "query": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
    "document": "",
    # Fixed per-task instructions (matches Hub config.json prompt_config)
    "AmazonCounterfactualClassification": "Instruct: Classify a given Amazon customer review text as either counterfactual or not counterfactual.\nQuery: ",
    "AmazonCounterfactualClassification-document": "",
    "AmazonReviewsClassification": "Instruct: Classify the given Amazon review into its appropriate rating category.\nQuery: ",
    "AmazonReviewsClassification-document": "",
    "BelebeleRetrieval": "Instruct: Given a question, retrieve a passage in any language that answers the question.\nQuery: ",
    "BelebeleRetrieval-document": "",
    "BlurbsClusteringP2P": "Instruct: Classify the given book title and blurb into its genre.\nQuery: ",
    "BlurbsClusteringS2S": "Instruct: Classify the given book title into its genre.\nQuery: ",
    "FalseFriendsGermanEnglish": "Instruct: Retrieve translations of the following text.\nQuery: ",
    "FloresBitextMining": "Instruct: Retrieve translation-equivalent text.\nQuery: ",
    "GerDaLIR": "Instruct: Given a legal query or passage, retrieve the court decision that addresses it.\nQuery: ",
    "GerDaLIR-document": "",
    "GermanSTSBenchmark": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "MLQARetrieval": "Instruct: Given a question, retrieve a passage in any language that answers the question.\nQuery: ",
    "MLQARetrieval-document": "",
    "MTOPDomainClassification": "Instruct: Classify the intent domain of the given utterance in task-oriented conversation.\nQuery: ",
    "MTOPDomainClassification-document": "",
    "MTOPIntentClassification": "Instruct: Classify the intent of the given utterance in task-oriented conversation.\nQuery: ",
    "MTOPIntentClassification-document": "",
    "MassiveIntentClassification": "Instruct: Classify the intent of the given utterance in task-oriented conversation.\nQuery: ",
    "MassiveIntentClassification-document": "",
    "MassiveScenarioClassification": "Instruct: Classify the scenario of the given utterance in task-oriented conversation.\nQuery: ",
    "MassiveScenarioClassification-document": "",
    "MultiEURLEXMultilabelClassification": "Instruct: Classify the given EU legal act into its EuroVoc subject domains.\nQuery: ",
    "MultiEURLEXMultilabelClassification-document": "",
    "MultiHateClassification": "Instruct: Classify the given text as either hateful or not hateful.\nQuery: ",
    "MultiHateClassification-document": "",
    "NTREXBitextMining": "Instruct: Retrieve translation-equivalent text.\nQuery: ",
    "RTE3": "Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise.\nQuery: ",
    "STS17": "Instruct: Retrieve semantically similar text in English or German.\nQuery: ",
    "STS22.v2": "Instruct: Retrieve semantically similar text.\nQuery: ",
    "SwissJudgementClassification": "Instruct: Classify the given facts of a Swiss court case by whether the appeal was approved or dismissed.\nQuery: ",
    "SwissJudgementClassification-document": "",
    "TenKGnadClusteringP2P": "Instruct: Classify the topic of the given text.\nQuery: ",
    "TenKGnadClusteringS2S": "Instruct: Classify the topic of the given text.\nQuery: ",
    "XMarket": "Instruct: Given a product name search, retrieve the corresponding product description.\nQuery: ",
    "XMarket-document": "",
    "XNLI": "Instruct: Given a premise, retrieve a hypothesis that is entailed by the premise.\nQuery: ",
}


def _mean_max_sim(
    queries: list[torch.Tensor], documents: list[torch.Tensor]
) -> torch.Tensor:
    """MaxSim averaged over each query's tokens."""
    import torch

    scores = max_sim(queries, documents)
    return scores / torch.tensor([len(q) for q in queries])[:, None].to(scores)


def _best_window_scores(
    query_dense: torch.Tensor,
    query_tokens: list[torch.Tensor],
    documents: list[dict[str, Any]],
) -> torch.Tensor:
    """Each query's best dense and best MaxSim window score per document, stacked."""
    import torch

    window_dense = torch.cat([d["window_dense"] for d in documents])
    window_tokens = [tokens for d in documents for tokens in d["window_tokens"]]
    owner = torch.tensor(
        [index for index, d in enumerate(documents) for _ in d["window_tokens"]]
    )
    best = torch.full((2, len(query_tokens), len(documents)), -torch.inf)
    # Blocks of windows keep the queries x windows score matrix small.
    for start in range(0, len(window_tokens), _WINDOW_BLOCK):
        block = slice(start, start + _WINDOW_BLOCK)
        block_owner = owner[block].expand(len(query_tokens), -1)
        block_scores = (
            dot_score(query_dense, window_dense[block]),
            _mean_max_sim(query_tokens, window_tokens[block]),
        )
        for kind, scores in enumerate(block_scores):
            best[kind] = best[kind].scatter_reduce(1, block_owner, scores, "amax")
    return best


def _hybrid_scores(queries: Array, documents: list[dict[str, Any]]) -> torch.Tensor:
    """Hybrid dense + MaxSim score of every query against every document."""
    import torch

    queries = [torch.as_tensor(query) for query in queries]
    dense_rows = documents[0]["dense"].shape[-1] // queries[0].shape[-1]
    query_dense = torch.stack([q[:dense_rows].reshape(-1) for q in queries])
    query_tokens = [q[dense_rows:] for q in queries]
    dense = dot_score(query_dense, torch.stack([d["dense"] for d in documents]))
    maxsim = _mean_max_sim(query_tokens, [d["tokens"] for d in documents])
    dense_window, maxsim_window = _best_window_scores(
        query_dense, query_tokens, documents
    )
    dense, maxsim = (dense + dense_window) / 2, (maxsim + maxsim_window) / 2
    return _DENSE_WEIGHT * dense + (1 - _DENSE_WEIGHT) * maxsim


class Hyper3EmbedWrapper(AbsEncoder):
    """Transformers encoder for ``hyper3-embed-deu`` with optional hybrid dense + MaxSim retrieval.

    Args:
        model: Model name or local path loaded via ``AutoModel.from_pretrained``.
        revision: Hub revision hash.
        device: Target torch device (defaults to CUDA when available, else CPU).
        model_prompts: Task-to-prompt mapping.
        embed_dim: Must be the model's embedding dimension (1,536) if given.
        multivector: When ``False`` (default), all tasks use 1,536-d dense vectors and cosine
            similarity. When ``True``, retrieval and reranking tasks score each query-document
            pair as ``0.25 * dense + 0.75 * MaxSim``, where each part is the mean of its score on
            the full document and on the best 512-token window; other tasks keep the dense
            vectors. The 128-d token vectors of every document in a corpus chunk are kept in
            memory, so long-document corpora need much more RAM than the default.
        **kwargs: Additional keyword arguments forwarded to ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        *,
        embed_dim: int | None = None,
        multivector: bool = False,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel

        self._mteb_model_meta: ModelMeta | None = None
        self.multivector = multivector
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModel.from_pretrained(model, revision=revision, **kwargs)
            .to(self.device)
            .eval()
        )
        if embed_dim is not None and embed_dim != self.model.embedding_dim:
            raise ValueError(
                f"embed_dim must be {self.model.embedding_dim}, got {embed_dim}."
            )
        self.model_prompts = self.validate_task_to_prompt_name(model_prompts)

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """Metadata reflecting whether hybrid multi-vector retrieval scoring is enabled."""
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta | None) -> None:
        if meta is not None and self.multivector:
            meta = meta.model_copy(
                update={"model_type": ["dense", "late-interaction"]}, deep=True
            )
        elif meta is not None:
            # An explicit multivector=False is the default model: keep it out of experiments/.
            experiment_kwargs = dict(meta.experiment_kwargs or {})
            experiment_kwargs.pop("multivector", None)
            meta = meta.model_copy(
                update={"experiment_kwargs": experiment_kwargs or None}, deep=True
            )
        self._mteb_model_meta = meta

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Encode dense vectors, or multi-vector representations for retrieval with ``multivector=True``."""
        texts = [text for batch in inputs for text in batch["text"]]
        prompt = get_prompt(self.model_prompts, task_metadata, prompt_type) or ""
        batch_size = kwargs.get("batch_size", 32)
        if (
            not self.multivector
            or task_metadata.simplified_task_type != "retrieval"
            or prompt_type is None
        ):
            return self.model.encode(texts, prompt=prompt, batch_size=batch_size)
        if prompt_type == PromptType.query:
            return self._encode_queries(texts, prompt, batch_size)
        return self._encode_documents(texts, prompt, batch_size)

    def _encode_queries(
        self, texts: list[str], prompt: str, batch_size: int
    ) -> list[torch.Tensor]:
        import torch

        dense, tokens = self.model.encode_with_tokens(
            texts, prompt=prompt, batch_size=batch_size, max_length=_QUERY_MAX_LENGTH
        )
        # One (12 + n_tokens, 128) tensor per query: the 1,536-d dense vector as 12 leading rows,
        # then the token vectors. SearchEncoderWrapper needs a single tensor per query to rerank.
        return [
            torch.cat((d.view(-1, t.shape[-1]), t.float()))
            for d, t in zip(dense, tokens, strict=True)
        ]

    def _encode_documents(
        self, texts: list[str], prompt: str, batch_size: int
    ) -> list[dict[str, Any]]:
        """Encode each document's full text and its 512-token windows as dense and token vectors."""
        import torch

        encode = self.model.encode_with_tokens
        dense, tokens = encode(texts, prompt=prompt, batch_size=batch_size)
        windows, owner = self.model.document_windows(texts)
        window_dense, window_tokens = encode(
            windows, prompt=prompt, batch_size=batch_size, max_length=_WINDOW_MAX_LENGTH
        )
        ends = torch.tensor(owner).bincount(minlength=len(texts)).cumsum(0).tolist()
        return [
            {
                "dense": dense[index],
                "tokens": tokens[index],
                "window_dense": window_dense[start:end],
                "window_tokens": window_tokens[start:end],
            }
            for index, (start, end) in enumerate(zip([0, *ends], ends, strict=False))
        ]

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Hybrid dense + MaxSim scores for multi-vector documents, otherwise cosine similarity."""
        if (
            isinstance(embeddings2, list)
            and embeddings2
            and isinstance(embeddings2[0], dict)
        ):
            return _hybrid_scores(embeddings1, embeddings2)
        return super().similarity(embeddings1, embeddings2)


# Train splits of these tasks, or data derived from them (translations, synthetic rows seeded from
# them, or shared underlying corpora and dataset families), were used in training.
HYPER3_EMBED_DEU_TRAINING_DATASETS: set[str] = {
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BQ",
    "Banking77Classification",
    "BiorxivClusteringS2S",
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "CEDRClassification",
    "CMedQAv2-reranking",
    "COIRCodeSearchNetRetrieval",
    "CmedqaRetrieval",
    "Cmnli",
    "CodeFeedbackMT",
    "CodeFeedbackST",
    "CodeSearchNetCCRetrieval",
    "CodeSearchNetRetrieval",
    "CosQA",
    "DuRetrieval",
    "ESCIReranking",
    "EcomRetrieval",
    "EmotionClassification",
    "FEVER",
    "FEVER-NL",
    "GerDaLIR",
    "GerDaLIRSmall",
    "GermanHealthcare1Retrieval",
    "GermanLegal1Retrieval",
    "GermanSTSBenchmark",
    "HotpotQA",
    "HotpotQA-NL",
    "ImdbClassification",
    "LCQMC",
    "LMEBMLDR",
    "LegalQuAD",
    "MIRACLReranking",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MKQARetrieval",
    "MLQARetrieval",
    "MLSUMClusteringP2P",
    "MLSUMClusteringP2P.v2",
    "MLSUMClusteringS2S",
    "MLSUMClusteringS2S.v2",
    "MMarcoReranking",
    "MMarcoRetrieval",
    "MMarcoRetrievalMultilingual",
    "MSMARCO",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MedicalQARetrieval",
    "MedicalRetrieval",
    "MrTidyRetrieval",
    "MultiEURLEXMultilabelClassification",
    "MultiLongDocReranking",
    "MultiLongDocRetrieval",
    "NQ",
    "Ocnli",
    "PAWSX",
    "PawsXPairClassification",
    "QBQTC",
    "QuoraRetrieval",
    "RTE3",
    "RedditClustering",
    "RedditClusteringP2P",
    "STS12",
    "STS17",
    "STS22",
    "STS22.v2",
    "STSBenchmark",
    "STSBenchmarkMultilingualSTS",
    "StackExchangeClustering",
    "StackOverflowDupQuestions",
    "SwissJudgementClassification",
    "SyntheticText2SQL",
    "T2Reranking",
    "T2Retrieval",
    "TNews",
    "TRECCOVID",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "TwentyNewsgroupsClustering",
    "Waimai",
    "WebFAQBitextMiningQAs",
    "WebFAQBitextMiningQuestions",
    "WebFAQRetrieval",
    "XMarket",
    "XNLI",
    "XNLIV2",
    "XQuADRetrieval",
    "YahooAnswersTopicsClassification",
    "mMARCO-NL",
}

hyper3_embed_deu = ModelMeta(
    loader=Hyper3EmbedWrapper,
    loader_kwargs=dict(trust_remote_code=True, model_prompts=HYPER3_EMBED_DEU_PROMPTS),
    name="hyper3labs/hyper3-embed-deu",
    revision="88b959f6b49424e5a3fd9fd5f776f52de5e0a037",
    release_date="2026-11-27",
    languages=["deu-Latn", "eng-Latn"],
    n_parameters=4_663_588_608,
    n_embedding_parameters=2_751_463_424,
    memory_usage_mb=8962,
    max_tokens=4096,
    embed_dim=1536,
    license="openmdw-1.1",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/hyper3labs/hyper3-embed-deu",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=HYPER3_EMBED_DEU_TRAINING_DATASETS,
    adapted_from="google/gemma-4-E2B-it",
    modalities=["text"],
    model_type=["dense"],
    extra_requirements_groups=["hyper3-embed"],
    contacts=["mnm-matin"],
    output_dtypes=OutputDType.FLOAT32,
)
