from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version

import mteb
from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

logger = logging.getLogger(__name__)

MODERN_BERT_TRANSFORMERS_MIN_VERSION = "4.48.0"


class NomicWrapper(SentenceTransformerWrapper):
    """following the hf model card documentation."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        if model_name == "nomic-ai/modernbert-embed-base" and (
            Version(transformers.__version__).release
            < Version(MODERN_BERT_TRANSFORMERS_MIN_VERSION).release
        ):
            raise RuntimeError(
                f"Current transformers version is {transformers.__version__} is lower than the required version"
                f" {MODERN_BERT_TRANSFORMERS_MIN_VERSION}"
            )
        super().__init__(model_name, revision, model_prompts, **kwargs)

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(  # type: ignore
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        # default to search_document if input_type and prompt_name are not provided
        prompt_name = (
            self.get_prompt_name(self.model_prompts, task_name, prompt_type)
            or PromptType.passage.value
        )
        task = mteb.get_task(task_name)
        # normalization not applied to classification
        # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/eval/mteb_eval/eval_mteb.py#L172
        normalize = task.metadata.type not in (
            "Classification",
            "MultilabelClassification",
            "PairClassification",
            "Reranking",
            "STS",
            "Summarization",
        )
        emb = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            batch_size=batch_size,
            **kwargs,
        )
        # v1.5 has a non-trainable layer norm to unit normalize the embeddings for binary quantization
        # the outputs are similar to if we just normalized but keeping the same for consistency
        if self.model_name == "nomic-ai/nomic-embed-text-v1.5":
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().float().numpy()
        return emb


nomic_training_data = {
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/data/contrastive_pretrain.yaml
    # reddit_title_body
    "RedditClustering": [],
    "RedditClusteringP2P": [],
    "RedditClustering.v2": [],
    "RedditClusteringP2P.v2": [],
    # amazon_reviews
    # amazonqa
    "AmazonPolarityClassification": [],
    "AmazonReviewsClassification": [],
    "AmazonCounterfactualClassification": [],
    # paq
    # s2orc_citation_titles
    # s2orc_title_abstract
    # s2orc_abstract_citation
    # s2orc_abstract_body
    # wikianswers
    # wikipedia
    "WikipediaRetrievalMultilingual": [],
    "WikipediaRerankingMultilingual": [],
    # gooaq
    # codesearch
    "CodeSearchNetCCRetrieval": [],
    "COIRCodeSearchNetRetrieval": [],
    # yahoo_title_answer
    # yahoo_qa
    # yahoo_title_question
    "YahooAnswersTopicsClassification": [],
    # agnews
    # ccnews
    # npr
    # eli5
    # cnn
    # stackexchange_duplicate_questions
    # stackexchange_title_body
    # stackexchange_body_body
    "StackExchangeClustering.v2": [],
    "StackExchangeClusteringP2P.v2": [],
    # sentence_compression
    # wikihow
    # altlex
    # quora
    "QuoraRetrieval": [],
    "Quora-NL": [],  # translation not trained on
    "NanoQuoraRetrieval": [],
    # simplewiki
    # squad
    "FQuADRetrieval": [],
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/data/finetune_triplets.yaml
    # msmaro
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "mMARCO-NL": ["train"],
    # nq_triples
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
    # nli_triplets
    # reddit
    # medi_wiki
    # medi_stackexchange
    # medi_flickr
    # medi_supernli
    # hotpot
    "HotPotQA": ["test"],
    "HotPotQAHardNegatives": ["test"],
    "HotPotQA-PL": ["test"],  # translated from hotpotQA (not trained on)
    "HotpotQA-NL": ["test"],  # translated from hotpotQA (not trained on)
    # fever
    "FEVER": ["test"],
    "FEVERHardNegatives": ["test"],
    "FEVER-NL": ["test"],  # translated, not trained on
}

# https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/eval/mteb_eval/eval_mteb.py#L142-L159
model_prompts = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.passage.value: "search_document: ",
}

nomic_embed_v1_5 = ModelMeta(
    loader=partial(
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        revision="b0753ae76394dd36bcfb912a46018088bca48be0",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    release_date="2024-02-10",  # first commit
    n_parameters=137_000_000,
    memory_usage_mb=522,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_data=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
)

nomic_embed_v1 = ModelMeta(
    loader=partial(
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1",
        revision="0759316f275aa0cb93a5b830973843ca66babcf5",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1",
    languages=["eng-Latn"],
    open_weights=True,
    revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    release_date="2024-01-31",  # first commit
    n_parameters=None,
    memory_usage_mb=522,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by="nomic-ai/nomic-embed-text-v1.5",
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_embed_v1_ablated = ModelMeta(
    loader=partial(
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1-ablated",
        revision="7d948905c5d5d3874fa55a925d68e49dbf411e5f",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-ablated",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7d948905c5d5d3874fa55a925d68e49dbf411e5f",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-ablated",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_embed_v1_unsupervised = ModelMeta(
    loader=partial(
        NomicWrapper,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1-unsupervised",
        revision="b53d557b15ae63852847c222d336c1609eced93c",
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-unsupervised",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b53d557b15ae63852847c222d336c1609eced93c",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_modern_bert_embed = ModelMeta(
    loader=partial(
        NomicWrapper,
        model_name="nomic-ai/modernbert-embed-base",
        revision="5960f1566fb7cb1adf1eb6e816639cf4646d9b12",
        model_prompts=model_prompts,
        model_kwargs={
            "torch_dtype": torch.float16,
        },
    ),
    name="nomic-ai/modernbert-embed-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="5960f1566fb7cb1adf1eb6e816639cf4646d9b12",
    release_date="2024-12-29",
    n_parameters=149_000_000,
    memory_usage_mb=568,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/modernbert-embed-base",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="answerdotai/ModernBERT-base",
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_pretrain_modernbert.yaml",
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune_modernnomic.yaml
    superseded_by=None,
    training_datasets=nomic_training_data,
    public_training_data=None,
)
