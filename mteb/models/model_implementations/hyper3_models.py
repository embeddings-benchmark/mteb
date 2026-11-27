"""hyper3-embed-deu by hyper3labs: German text embeddings fine-tuned from Gemma 4 E2B.

Loads via Sentence Transformers (``trust_remote_code=True``): the Hub repository's custom modules
restore the bidirectional sliding window, append EOS, and pool with an FP32 latent cross-attention
block (excluding prompt tokens from the mean).

By default (``multivector=False``), ``Hyper3EmbedWrapper`` encodes 1,536-d normalized FP32
embeddings (up to 4,096 tokens) scored with cosine similarity. Passing ``multivector=True`` enables
the model's fused dense + 128-d MaxSim retrieval scorer over full documents and overlapping
512-token windows (stride 256).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
    _resolve_prompt,
)
from mteb.types import OutputDType, PromptType

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

    class _Hyper3CoreModel(Protocol):
        embedding_dim: int
        backbone: torch.nn.Module
        multivec: torch.nn.Linear | None

        def encode(
            self,
            texts: str | list[str],
            prompt: str | None = None,
            prompt_name: str | None = None,
            batch_size: int = 8,
            max_length: int | None = None,
        ) -> torch.Tensor: ...

        def encode_with_tokens(
            self,
            texts: str | list[str],
            prompt: str | None = None,
            prompt_name: str | None = None,
            batch_size: int = 8,
            max_length: int | None = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

        def _windows(
            self, texts: list[str], window: int, stride: int
        ) -> tuple[list[str], list[int], list[int]]: ...

        def maxsim_scores(
            self,
            queries: list[torch.Tensor],
            documents: list[torch.Tensor],
            query_block: int = 256,
            budget: int | None = None,
        ) -> torch.Tensor: ...

        def fuse_scores(
            self,
            dense_full: torch.Tensor,
            dense_window: torch.Tensor,
            maxsim_full: torch.Tensor | None,
            maxsim_window: torch.Tensor,
            document_tokens: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> torch.Tensor: ...


_QUERY_MAX_LENGTH = 1024
_CHUNK_WINDOW = 512
_CHUNK_STRIDE = 256
_DOC_BLOCK = 512
_WINDOW_BLOCK = 4096

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


@dataclass(frozen=True, slots=True)
class _DocumentEmbeddings:
    """CPU-resident full-document and 512-token window representations for fused scoring."""

    dense: torch.Tensor
    tokens: torch.Tensor | None
    window_dense: torch.Tensor
    window_tokens: list[torch.Tensor]
    num_tokens: int


def _is_multivector_documents(
    embeddings: Array | Sequence[_DocumentEmbeddings],
) -> TypeGuard[Sequence[_DocumentEmbeddings]]:
    return (
        isinstance(embeddings, Sequence)
        and bool(embeddings)
        and isinstance(embeddings[0], _DocumentEmbeddings)
    )


def _as_query_list(queries: Array) -> list[torch.Tensor]:
    import torch

    if isinstance(queries, torch.Tensor) and queries.ndim == 2:
        return [queries]
    if isinstance(queries, torch.Tensor) and queries.ndim == 3:
        return [queries[i] for i in range(queries.shape[0])]
    return [torch.as_tensor(q) for q in queries]


class Hyper3EmbedWrapper(SentenceTransformerEncoderWrapper):
    """Sentence Transformers encoder with optional fused dense + MaxSim retrieval scoring.

    Args:
        model: Model name, path, or loaded ``SentenceTransformer`` instance.
        revision: Hub revision hash.
        device: Target torch device.
        model_prompts: Task-to-prompt mapping passed to ``SentenceTransformerEncoderWrapper``.
        multivector: When ``False`` (default), encodes 1,536-d dense vectors for all tasks.
            When ``True``, retrieval and reranking tasks encode full-document and 512-token
            window representations and score them with the checkpoint's fused dense + 128-d
            MaxSim scorer; non-retrieval tasks continue to use the 1,536-d dense embeddings.
        fusion: Optional keyword overrides forwarded to ``fuse_scores`` when ``multivector=True``.
        **kwargs: Additional keyword arguments forwarded to ``SentenceTransformerEncoderWrapper``.
    """

    def __init__(
        self,
        model: str | SentenceTransformer,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        *,
        multivector: bool = False,
        fusion: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not multivector and fusion is not None:
            raise ValueError("fusion requires multivector=True")
        self.multivector = multivector
        self.fusion = fusion
        super().__init__(
            model,
            revision=revision,
            device=device,
            model_prompts=model_prompts,
            **kwargs,
        )
        if multivector and self._core_model.multivec is None:
            raise ValueError("This checkpoint has no MaxSim head.")

    @property
    def _core_model(self) -> _Hyper3CoreModel:
        return cast("_Hyper3CoreModel", self.model[0].model)

    def _use_full_maxsim(self) -> bool:
        if self.fusion is None:
            return True
        return bool(self.fusion.get("maxsim_window_weight", 0.5) != 1)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        """Metadata reflecting whether fused multi-vector retrieval scoring is enabled."""
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta) -> None:
        experiment_kwargs = dict(meta.experiment_kwargs or {})
        if not self.multivector:
            experiment_kwargs.pop("multivector", None)
            experiment_kwargs.pop("fusion", None)
            self._mteb_model_meta = meta.model_copy(
                update={"experiment_kwargs": experiment_kwargs or None},
                deep=True,
            )
            return
        self._mteb_model_meta = meta.model_copy(
            update={
                "model_type": ["dense", "late-interaction"],
                "experiment_kwargs": experiment_kwargs or None,
            },
            deep=True,
        )

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
        """Encode inputs as 1,536-d dense vectors or multi-vector retrieval representations."""
        if (
            not self.multivector
            or task_metadata.simplified_task_type != "retrieval"
            or prompt_type is None
        ):
            return super().encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )

        texts = [text for batch in inputs for text in batch["text"]]
        prompt = _resolve_prompt(self.model_prompts, task_metadata, prompt_type) or ""
        batch_size = kwargs.get("batch_size", 8)
        if prompt_type == PromptType.query:
            return self._encode_queries(texts, prompt=prompt, batch_size=batch_size)
        return self._encode_documents(texts, prompt=prompt, batch_size=batch_size)

    def _encode_queries(
        self, texts: list[str], *, prompt: str, batch_size: int
    ) -> Array:
        import torch

        if not texts:
            return cast("Array", [])
        core = self._core_model
        multivec = cast("torch.nn.Linear", core.multivec)
        token_dim = int(multivec.out_features)
        if core.embedding_dim % token_dim != 0:
            raise ValueError(
                f"embedding_dim ({core.embedding_dim}) must be divisible by multivec dim ({token_dim})"
            )
        dense_rows = core.embedding_dim // token_dim
        query_dense, query_tokens = core.encode_with_tokens(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            max_length=_QUERY_MAX_LENGTH,
        )
        # Pack the 1,536-d dense vector as the leading rows of a 2D (12 + L_q, 128) tensor so
        # SearchEncoderWrapper's `torch.as_tensor(query_embeddings[q_idx]).unsqueeze(0)` works
        # unchanged on reranking tasks.
        return cast(
            "Array",
            [
                torch.cat((dense.view(dense_rows, token_dim), tokens.float()), dim=0)
                for dense, tokens in zip(query_dense, query_tokens, strict=True)
            ],
        )

    def _encode_full_documents(
        self, texts: list[str], *, prompt: str, batch_size: int
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        import torch

        core = self._core_model
        full_maxsim = self._use_full_maxsim()
        dense_parts: list[torch.Tensor] = []
        tokens: list[torch.Tensor | None] = []
        for start in range(0, len(texts), _DOC_BLOCK):
            block = texts[start : start + _DOC_BLOCK]
            if full_maxsim:
                dd, dt = core.encode_with_tokens(
                    block, prompt=prompt, batch_size=batch_size
                )
                dense_parts.append(dd)
                tokens.extend(dt)
            else:
                dense_parts.append(
                    core.encode(block, prompt=prompt, batch_size=batch_size)
                )
                tokens.extend([None] * len(block))
        return torch.cat(dense_parts, dim=0), tokens

    def _encode_document_windows(
        self, texts: list[str], *, prompt: str, batch_size: int
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int], list[int]]:
        import torch

        core = self._core_model
        windows, owner, lengths = core._windows(texts, _CHUNK_WINDOW, _CHUNK_STRIDE)
        dense_parts: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        for start in range(0, len(windows), _WINDOW_BLOCK):
            wd, wt = core.encode_with_tokens(
                windows[start : start + _WINDOW_BLOCK],
                prompt=prompt,
                batch_size=batch_size,
                max_length=_CHUNK_WINDOW + 64,
            )
            dense_parts.append(wd)
            tokens.extend(wt)
        win_counts = [0] * len(texts)
        for doc_idx in owner:
            win_counts[doc_idx] += 1
        return torch.cat(dense_parts, dim=0), tokens, win_counts, lengths

    def _encode_documents(
        self, texts: list[str], *, prompt: str, batch_size: int
    ) -> Array:
        if not texts:
            return cast("Array", [])
        doc_dense, doc_tokens = self._encode_full_documents(
            texts, prompt=prompt, batch_size=batch_size
        )
        win_dense, win_tokens, win_counts, lengths = self._encode_document_windows(
            texts, prompt=prompt, batch_size=batch_size
        )
        docs: list[_DocumentEmbeddings] = []
        offset = 0
        for idx, count in enumerate(win_counts):
            end = offset + count
            docs.append(
                _DocumentEmbeddings(
                    dense=doc_dense[idx],
                    tokens=doc_tokens[idx],
                    window_dense=win_dense[offset:end],
                    window_tokens=win_tokens[offset:end],
                    num_tokens=lengths[idx],
                )
            )
            offset = end
        return cast("Array", docs)

    def _score_full_documents(
        self,
        qd: torch.Tensor,
        qt: list[torch.Tensor],
        documents: Sequence[_DocumentEmbeddings],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        import torch

        core = self._core_model
        n_queries, n_docs = len(qt), len(documents)
        dense_full = torch.empty(
            (n_queries, n_docs), dtype=torch.float32, device=qd.device
        )
        maxsim_full = (
            torch.empty((n_queries, n_docs), dtype=torch.float32, device=qd.device)
            if self._use_full_maxsim()
            else None
        )
        for start in range(0, n_docs, _DOC_BLOCK):
            block = documents[start : start + _DOC_BLOCK]
            dd = torch.stack([doc.dense for doc in block]).to(
                device=qd.device, dtype=torch.float32
            )
            dense_full[:, start : start + len(block)] = qd @ dd.T
            if maxsim_full is not None:
                dt = [cast("torch.Tensor", doc.tokens) for doc in block]
                maxsim_full[:, start : start + len(block)] = core.maxsim_scores(qt, dt)
        return dense_full, maxsim_full

    def _score_document_windows(
        self,
        qd: torch.Tensor,
        qt: list[torch.Tensor],
        documents: Sequence[_DocumentEmbeddings],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch

        core = self._core_model
        win_tokens: list[torch.Tensor] = []
        owner_list: list[int] = []
        for doc_idx, doc in enumerate(documents):
            win_tokens.extend(doc.window_tokens)
            owner_list.extend([doc_idx] * len(doc.window_tokens))

        all_wd = torch.cat([doc.window_dense for doc in documents], dim=0)
        owner = torch.tensor(owner_list, dtype=torch.long, device=qd.device)
        shape = (len(qt), len(documents))
        dense_window = torch.full(
            shape, -torch.inf, dtype=torch.float32, device=qd.device
        )
        maxsim_window = torch.full(
            shape, -torch.inf, dtype=torch.float32, device=qd.device
        )
        for start in range(0, len(win_tokens), _WINDOW_BLOCK):
            end = min(start + _WINDOW_BLOCK, len(win_tokens))
            wd = all_wd[start:end].to(device=qd.device, dtype=torch.float32)
            own = owner[start:end].expand(len(qt), -1)
            dense_window = dense_window.scatter_reduce(1, own, qd @ wd.T, "amax")
            maxsim_window = maxsim_window.scatter_reduce(
                1, own, core.maxsim_scores(qt, win_tokens[start:end]), "amax"
            )
        doc_tokens = torch.tensor(
            [doc.num_tokens for doc in documents], dtype=torch.long, device=qd.device
        )
        return dense_window, maxsim_window, doc_tokens

    def _score_multivector(
        self,
        queries: Array,
        documents: Sequence[_DocumentEmbeddings],
    ) -> torch.Tensor:
        import torch

        query_list = _as_query_list(queries)
        if not query_list or not documents:
            return torch.empty((len(query_list), len(documents)), dtype=torch.float32)

        core = self._core_model
        device = next(core.backbone.parameters()).device
        multivec = cast("torch.nn.Linear", core.multivec)
        dense_rows = core.embedding_dim // int(multivec.out_features)
        qd = torch.stack([q[:dense_rows].reshape(-1) for q in query_list]).to(
            device=device, dtype=torch.float32
        )
        qt = [q[dense_rows:].to(device=device, dtype=torch.float32) for q in query_list]

        dense_full, maxsim_full = self._score_full_documents(qd, qt, documents)
        dense_window, maxsim_window, doc_tokens = self._score_document_windows(
            qd, qt, documents
        )
        fused = core.fuse_scores(
            dense_full=dense_full,
            dense_window=dense_window,
            maxsim_full=maxsim_full,
            maxsim_window=maxsim_window,
            document_tokens=doc_tokens,
            **(self.fusion or {}),
        )
        return fused.float().cpu()

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score retrieval pairs with fused dense + MaxSim when multi-vector, else cosine."""
        if _is_multivector_documents(embeddings2):
            return self._score_multivector(embeddings1, embeddings2)
        if isinstance(embeddings2, Sequence) and not embeddings2:
            import torch

            return torch.empty((len(embeddings1), 0), dtype=torch.float32)
        return super().similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Score aligned pairs with fused dense + MaxSim when multi-vector, else cosine."""
        if _is_multivector_documents(embeddings2):
            import torch

            queries = _as_query_list(embeddings1)
            return torch.cat(
                [
                    self._score_multivector(cast("Array", [q]), [doc]).squeeze(0)
                    for q, doc in zip(queries, embeddings2, strict=True)
                ]
            )
        if isinstance(embeddings2, Sequence) and not embeddings2:
            import torch

            return torch.empty((0,), dtype=torch.float32)
        return super().similarity_pairwise(embeddings1, embeddings2)


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
