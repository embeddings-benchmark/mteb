from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from packaging.version import Version

from mteb.models import sentence_transformers_loader
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

MODERN_BERT_TRANSFORMERS_MIN_VERSION = "4.48.0"


class NomicWrapper(SentenceTransformerEncoderWrapper):
    """following the hf model card documentation."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        import transformers

        self.model_name = model_name
        if model_name == "nomic-ai/modernbert-embed-base" and (
            Version(transformers.__version__).release
            < Version(MODERN_BERT_TRANSFORMERS_MIN_VERSION).release
        ):
            raise RuntimeError(
                f"Current transformers version is {transformers.__version__} is lower than the required version"
                f" {MODERN_BERT_TRANSFORMERS_MIN_VERSION}"
            )
        super().__init__(
            model_name, revision, device=device, model_prompts=model_prompts, **kwargs
        )

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        # default to search_document if input_type and prompt_name are not provided
        prompt_name = (
            self.get_prompt_name(task_metadata, prompt_type)
            or PromptType.document.value
        )
        sentences = [text for batch in inputs for text in batch["text"]]

        # normalization not applied to classification
        # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/eval/mteb_eval/eval_mteb.py#L172
        normalize = task_metadata not in (
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
    "RedditClustering",
    "RedditClusteringP2P",
    "RedditClustering.v2",
    "RedditClusteringP2P.v2",
    # amazon_reviews
    # amazonqa
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    # paq
    # s2orc_citation_titles
    # s2orc_title_abstract
    # s2orc_abstract_citation
    # s2orc_abstract_body
    # wikianswers
    # wikipedia
    "WikipediaRetrievalMultilingual",
    "WikipediaRerankingMultilingual",
    # gooaq
    # codesearch
    "CodeSearchNetCCRetrieval",
    "COIRCodeSearchNetRetrieval",
    # yahoo_title_answer
    # yahoo_qa
    # yahoo_title_question
    "YahooAnswersTopicsClassification",
    # agnews
    # ccnews
    # npr
    # eli5
    # cnn
    # stackexchange_duplicate_questions
    # stackexchange_title_body
    # stackexchange_body_body
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P.v2",
    # sentence_compression
    # wikihow
    # altlex
    # quora
    "QuoraRetrieval",
    "Quora-NL",  # translation not trained on
    "NanoQuoraRetrieval",
    # simplewiki
    # squad
    "FQuADRetrieval",
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/data/finetune_triplets.yaml
    # msmaro
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "mMARCO-NL",
    # nq_triples
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "NQ-NL",  # translation not trained on
    # nli_triplets
    # reddit
    # medi_wiki
    # medi_stackexchange
    # medi_flickr
    # medi_supernli
    # hotpot
    "HotPotQA",
    "HotPotQAHardNegatives",
    "HotPotQA-PL",  # translated from hotpotQA (not trained on)
    "HotpotQA-NL",  # translated from hotpotQA (not trained on)
    # fever
    "FEVER",
    "FEVERHardNegatives",
    "FEVER-NL",  # translated, not trained on
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
    PromptType.document.value: "search_document: ",
}

NOMIC_CITATION = """
@misc{nussbaum2024nomic,
      title={Nomic Embed: Training a Reproducible Long Context Text Embedder},
      author={Zach Nussbaum and John X. Morris and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2402.01613},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

nomic_embed_v1_5 = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="b0753ae76394dd36bcfb912a46018088bca48be0",
    release_date="2024-02-10",  # first commit
    citation=NOMIC_CITATION,
    n_parameters=137_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=522,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_data=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
)

nomic_embed_v1 = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="0759316f275aa0cb93a5b830973843ca66babcf5",
    release_date="2024-01-31",  # first commit
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=522,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    use_instructions=True,
    citation=NOMIC_CITATION,
    adapted_from=None,
    superseded_by="nomic-ai/nomic-embed-text-v1.5",
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_embed_v1_ablated = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-ablated",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="7d948905c5d5d3874fa55a925d68e49dbf411e5f",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-ablated",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_embed_v1_unsupervised = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v1-unsupervised",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="b53d557b15ae63852847c222d336c1609eced93c",
    release_date="2024-01-15",  # first commit
    n_parameters=None,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "Transformers"],
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune.yaml",
    training_datasets=nomic_training_data,
    public_training_data=None,
)

nomic_modern_bert_embed = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/modernbert-embed-base",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="5960f1566fb7cb1adf1eb6e816639cf4646d9b12",
    release_date="2024-12-29",
    n_parameters=149_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=568,
    max_tokens=8192,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/modernbert-embed-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "ONNX", "safetensors"],
    use_instructions=True,
    adapted_from="answerdotai/ModernBERT-base",
    public_training_code="https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_pretrain_modernbert.yaml",
    # https://github.com/nomic-ai/contrastors/blob/5f7b461e5a13b5636692d1c9f1141b27232fe966/src/contrastors/configs/train/contrastive_finetune_modernnomic.yaml
    superseded_by=None,
    training_datasets=nomic_training_data,
    public_training_data=None,
    citation="""@misc{nussbaum2024nomic,
      title={Nomic Embed: Training a Reproducible Long Context Text Embedder},
      author={Zach Nussbaum and John X. Morris and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2402.01613},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
)


m_languages = [
    "eng-Latn",
    "spa-Latn",
    "fra-Latn",
    "deu-Latn",
    "ita-Latn",
    "por-Latn",
    "pol-Latn",
    "nld-Latn",
    "tur-Latn",
    "jpn-Jpan",
    "vie-Latn",
    "rus-Cyrl",
    "ind-Latn",
    "arb-Arab",
    "ces-Latn",
    "ron-Latn",
    "swe-Latn",
    "ell-Grek",
    "ukr-Cyrl",
    "zho-Hans",
    "hun-Latn",
    "dan-Latn",
    "nor-Latn",
    "hin-Deva",
    "fin-Latn",
    "bul-Cyrl",
    "kor-Hang",
    "slk-Latn",
    "tha-Thai",
    "heb-Hebr",
    "cat-Latn",
    "lit-Latn",
    "fas-Arab",
    "msa-Latn",
    "slv-Latn",
    "lav-Latn",
    "mar-Deva",
    "ben-Beng",
    "sqi-Latn",
    "cym-Latn",
    "bel-Cyrl",
    "mal-Mlym",
    "kan-Knda",
    "mkd-Cyrl",
    "urd-Arab",
    "fry-Latn",
    "fil-Latn",
    "tel-Telu",
    "eus-Latn",
    "swh-Latn",
    "som-Latn",
    "snd-Arab",
    "uzb-Latn",
    "cos-Latn",
    "hrv-Latn",
    "guj-Gujr",
    "hin-Latn",
    "ceb-Latn",
    "epo-Latn",
    "jav-Latn",
    "lat-Latn",
    "zul-Latn",
    "mon-Cyrl",
    "sin-Sinh",
    "ell-Latn",
    "gle-Latn",
    "kir-Cyrl",
    "tgk-Cyrl",
    "mya-Mymr",
    "khm-Khmr",
    "mlg-Latn",
    "pan-Guru",
    "rus-Latn",
    "sna-Latn",
    "zho-Latn",
    "hau-Latn",
    "heb-Latn",
    "hmn-Latn",
    "hat-Latn",
    "jpn-Latn",
    "sun-Latn",
    "bul-Latn",
    "gla-Latn",
    "nya-Latn",
    "pus-Arab",
    "kur-Latn",
    "hbs-Latn",
    "amh-Ethi",
    "ibo-Latn",
    "lao-Laoo",
    "mri-Latn",
    "nno-Latn",
    "smo-Latn",
    "yid-Hebr",
    "sot-Latn",
    "tgl-Latn",
    "xho-Latn",
    "yor-Latn",
]

nomic_embed_text_v2_moe = ModelMeta(
    loader=NomicWrapper,
    loader_kwargs=dict(
        trust_remote_code=True,
        model_prompts=model_prompts,
    ),
    name="nomic-ai/nomic-embed-text-v2-moe",
    model_type=["dense"],
    languages=m_languages,
    open_weights=True,
    revision="1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
    release_date="2025-02-07",
    n_parameters=475292928,
    n_embedding_parameters=192036864,
    n_active_parameters_override=141628032,
    memory_usage_mb=1813,
    max_tokens=512,
    embed_dim=768,
    license="apache-2.0",
    reference="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    adapted_from="nomic-ai/nomic-xlm-2048",
    public_training_data="https://github.com/nomic-ai/contrastors?tab=readme-ov-file#data-access",
    public_training_code="https://github.com/nomic-ai/contrastors/blob/613ddfd37309e538cceadb05b1e6423e7b09f603/src/contrastors/configs/train/contrastive_finetune_moe.yaml",
    training_datasets=None,  # did not look into this further
    superseded_by=None,
    citation="""@misc{nussbaum2025trainingsparsemixtureexperts,
      title={Training Sparse Mixture Of Experts Text Embedding Models},
      author={Zach Nussbaum and Brandon Duderstadt},
      year={2025},
      eprint={2502.07972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07972},
}""",
)

nomic_embed_code = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs={
        "trust_remote_code": True,
        "model_prompts": model_prompts,
    },
    name="nomic-ai/nomic-embed-code",
    revision="11114029805cee545ef111d5144b623787462a52",
    release_date="2025-03-24",
    languages=["eng-Latn"],
    n_parameters=7_070_619_136,
    n_embedding_parameters=None,
    memory_usage_mb=26972.0,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/gangiswag/cornstack/",
    public_training_data="https://huggingface.co/collections/nomic-ai/cornstack",
    framework=["PyTorch", "Sentence Transformers", "safetensors"],
    reference="https://huggingface.co/nomic-ai/nomic-embed-code",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets={"CoRNStack"},
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation="""@misc{suresh2025cornstackhighqualitycontrastivedata,
      title={CoRNStack: High-Quality Contrastive Data for Better Code Retrieval and Reranking},
      author={Tarun Suresh and Revanth Gangi Reddy and Yifei Xu and Zach Nussbaum and Andriy Mulyar and Brandon Duderstadt and Heng Ji},
      year={2025},
      eprint={2412.01007},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.01007},
}""",
)
