"""hyper3-embed-deu by hyper3labs: German text embeddings fine-tuned from Gemma 4 E2B.

Loads via Sentence Transformers (``trust_remote_code=True``): the Hub repository's custom modules
restore the bidirectional sliding window, append EOS, and pool with an FP32 latent cross-attention
block (excluding prompt tokens from the mean).

By default (``multivector=False``), loads the standard dense ``SentenceTransformerEncoderWrapper``
(1,536-d normalized FP32 embeddings, up to 4,096 tokens). Passing ``multivector=True`` loads
``Hyper3MultiVectorWrapper``, which scores retrieval and reranking pairs with the repository's
fused dense + 128-d MaxSim scorer over full documents and overlapping 512-token windows (stride 256).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb._create_dataloaders import create_dataloader
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
    _resolve_prompt,
)
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.models_protocols import MTEBModels
    from mteb.types import (
        CorpusDatasetType,
        EncodeKwargs,
        QueryDatasetType,
        RetrievalOutputType,
        TopRankedDocumentsType,
    )

# Set both fields to the immutable public 40-hex Hub commit and release date via bind_public_sha.py.
HYPER3_EMBED_DEU_REVISION: str | None = "88b959f6b49424e5a3fd9fd5f776f52de5e0a037"
HYPER3_EMBED_DEU_RELEASE_DATE: str | None = "2026-11-27"

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


def _rank(
    results: RetrievalOutputType,
    scores: torch.Tensor,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    top_ranked: TopRankedDocumentsType | None,
) -> None:
    """Populate each query's top_k documents (within top_ranked candidates when given); -inf scores are dropped."""
    position = {doc_id: index for index, doc_id in enumerate(corpus_ids)}
    for row, query_id in enumerate(query_ids):
        if top_ranked is None:
            ranked, values = corpus_ids, scores[row]
        elif query_id in top_ranked:
            ranked = top_ranked[query_id]
            values = scores[row, [position[doc_id] for doc_id in ranked]]
        else:
            continue
        top = values.topk(min(top_k, len(ranked)))
        results[query_id] = {
            ranked[i]: v
            for v, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
            if v != float("-inf")
        }


class Hyper3MultiVectorWrapper(SentenceTransformerEncoderWrapper):
    """Sentence Transformers encoder; retrieval and reranking search with the repository's fused dense + MaxSim score."""

    def __init__(
        self,
        *args: Any,
        rerank_top_k: int | None = None,
        fusion: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.model[0].model.multivec is None:
            raise ValueError("this checkpoint has no MaxSim head")
        self.rerank_top_k, self.fusion = rerank_top_k, fusion

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None = None,
    ) -> None:
        loader = create_dataloader(
            corpus,
            task_metadata=task_metadata,
            prompt_type=PromptType.document,
            num_proc=num_proc,
            **encode_kwargs,
        )
        self._corpus_texts = [text for batch in loader for text in batch["text"]]
        self._corpus_ids = list(corpus["id"])

    def score_corpus(
        self,
        query_texts: list[str],
        document_texts: list[str],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
    ) -> torch.Tensor:
        return self.model[0].model.score_multivector(
            query_texts,
            document_texts,
            rerank_top_k=self.rerank_top_k,
            fusion=self.fusion,
            query_prompt=_resolve_prompt(
                self.model_prompts, task_metadata, PromptType.query
            )
            or "",
            document_prompt=_resolve_prompt(
                self.model_prompts, task_metadata, PromptType.document
            )
            or "",
            batch_size=encode_kwargs.get("batch_size", 8),
        )

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None = None,
    ) -> RetrievalOutputType:
        loader = create_dataloader(
            queries,
            task_metadata=task_metadata,
            prompt_type=PromptType.query,
            num_proc=num_proc,
            **encode_kwargs,
        )
        scores = self.score_corpus(
            [text for batch in loader for text in batch["text"]],
            self._corpus_texts,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            encode_kwargs=encode_kwargs,
        )
        results: RetrievalOutputType = {}
        _rank(results, scores, list(queries["id"]), self._corpus_ids, top_k, top_ranked)
        del self._corpus_texts, self._corpus_ids
        return results


class Hyper3ModelMeta(ModelMeta):
    """Updates ``model_type`` when loaded with ``multivector=True``."""

    def load_model(
        self,
        device: str | None = None,
        *,
        embed_dim: int | None = None,
        **kwargs: Any,
    ) -> MTEBModels:
        model = super().load_model(device=device, embed_dim=embed_dim, **kwargs)
        if kwargs.get("multivector", False):
            model.mteb_model_meta = model.mteb_model_meta.model_copy(
                update={"model_type": ["dense", "late-interaction"]}
            )
        return model


def hyper3_embed_deu_loader(
    model_name: str,
    revision: str | None = None,
    multivector: bool = False,
    rerank_top_k: int | None = None,
    fusion: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SentenceTransformerEncoderWrapper:
    """Default ``multivector=False`` loads dense ST; ``multivector=True`` loads ``Hyper3MultiVectorWrapper``."""
    if multivector:
        return Hyper3MultiVectorWrapper(
            model_name,
            revision=revision,
            rerank_top_k=rerank_top_k,
            fusion=fusion,
            **kwargs,
        )
    if rerank_top_k is not None or fusion is not None:
        raise ValueError("rerank_top_k and fusion require multivector=True")
    return SentenceTransformerEncoderWrapper(model_name, revision=revision, **kwargs)


# Train splits of these tasks, or data derived from them (translations, synthetic rows seeded from them,
# or shared underlying corpora), were used in training.
HYPER3_EMBED_DEU_TRAINING_DATASETS: set[str] = {
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BQ",
    "Banking77Classification",
    "BiorxivClusteringS2S",
    "COIRCodeSearchNetRetrieval",
    "Cmnli",
    "CodeFeedbackMT",
    "CodeFeedbackST",
    "CodeSearchNetCCRetrieval",
    "CodeSearchNetRetrieval",
    "CosQA",
    "ESCIReranking",
    "EmotionClassification",
    "FEVER",
    "FEVER-NL",
    "GerDaLIR",
    "GerDaLIRSmall",
    "GermanLegal1Retrieval",
    "GermanSTSBenchmark",
    "HotpotQA",
    "HotpotQA-NL",
    "ImdbClassification",
    "LCQMC",
    "LMEBMLDR",
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
    "MrTidyRetrieval",
    "MultiEURLEXMultilabelClassification",
    "MultiLongDocRetrieval",
    "NQ",
    "PAWSX",
    "PawsXPairClassification",
    "QBQTC",
    "QuoraRetrieval",
    "RTE3",
    "RedditClustering",
    "RedditClusteringP2P",
    "STS12",
    "STS22",
    "STS22.v2",
    "STSBenchmark",
    "STSBenchmarkMultilingualSTS",
    "StackExchangeClustering",
    "StackOverflowDupQuestions",
    "SwissJudgementClassification",
    "SyntheticText2SQL",
    "TNews",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "TwentyNewsgroupsClustering",
    "Waimai",
    "WebFAQRetrieval",
    "XMarket",
    "XNLI",
    "mMARCO-NL",
    # probable: same dataset family or corpus
    "BlurbsClusteringP2P",
    "BlurbsClusteringS2S",
    "CEDRClassification",
    "CMedQAv2-reranking",
    "CmedqaRetrieval",
    "DuRetrieval",
    "EcomRetrieval",
    "GermanHealthcare1Retrieval",
    "LegalQuAD",
    "MedicalQARetrieval",
    "MedicalRetrieval",
    "MultiLongDocReranking",
    "Ocnli",
    "STS17",
    "T2Reranking",
    "T2Retrieval",
    "TRECCOVID",
    "ThuNewsClusteringP2P",
    "ThuNewsClusteringS2S",
    "WebFAQBitextMiningQAs",
    "WebFAQBitextMiningQuestions",
    "XNLIV2",
    "XQuADRetrieval",
    "YahooAnswersTopicsClassification",
}

hyper3_embed_deu = Hyper3ModelMeta(
    loader=hyper3_embed_deu_loader,
    loader_kwargs=dict(trust_remote_code=True, model_prompts=HYPER3_EMBED_DEU_PROMPTS),
    name="hyper3labs/hyper3-embed-deu",
    revision=HYPER3_EMBED_DEU_REVISION,
    release_date=HYPER3_EMBED_DEU_RELEASE_DATE,
    languages=["deu-Latn", "eng-Latn"],
    n_parameters=4_663_588_608,
    n_active_parameters_override=None,
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
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    extra_requirements_groups=["hyper3-embed"],
    citation=None,
    contacts=["mnm-matin"],
    output_dtypes=OutputDType.FLOAT32,
)
