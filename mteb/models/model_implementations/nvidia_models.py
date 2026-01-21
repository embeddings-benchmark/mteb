from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from packaging.version import Version
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import __version__ as transformers_version

from mteb._requires_package import requires_package
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import CrossEncoderWrapper
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)

NV_RETRIEVER_CITATION = """@misc{lee2025nvembedimprovedtechniquestraining,
      title={NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models},
      author={Chankyu Lee and Rajarshi Roy and Mengyao Xu and Jonathan Raiman and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
      year={2025},
      eprint={2405.17428},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.17428},
}"""

LlamaEmbedNemotron_CITATION = """@misc{babakhin2025llamaembednemotron8buniversaltextembedding,
      title={Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks},
      author={Yauhen Babakhin and Radek Osmulski and Ronay Ak and Gabriel Moreira and Mengyao Xu and Benedikt Schifferer and Bo Liu and Even Oldridge},
      year={2025},
      eprint={2511.07025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07025},
}"""


def _instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return f"Instruct: {instruction}\nQuery: " if instruction else ""


nvidia_training_datasets = {
    # source: https://arxiv.org/pdf/2405.17428
    "ArguAna",
    "ArguAna-PL",
    "ArguAna-NL",  # translation not trained on
    "NanoArguAnaRetrieval",
    "HotpotQA",
    "HotpotQA-PL",  # translation not trained on
    "HotpotQA-NL",  # translation not trained on
    "HotpotQAHardNegatives",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "NanoMSMARCORetrieval",
    "MSMARCO-PL",  # translation not trained on
    "mMARCO-NL",  # translation not trained on
    "NQ",
    "NQHardNegatives",
    "NanoNQRetrieval",
    "NQ-PL",  # translation not trained on
    "NQ-NL",  # translation not trained on
    "FEVER",
    "FEVER-NL",  # translation not trained on
    "FEVERHardNegatives",
    "NanoFEVERRetrieval",
    "FiQA2018",
    "FiQA2018-NL",  # translation not trained on
    "STS12",
    "STS22",
    "AmazonReviewsClassification",
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "ArxivClusteringP2P",
    "ArxivClusteringP2P.v2",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringP2P.v2",
    "BiorxivClusteringS2S",
    "BiorxivClusteringS2S.v2",
    "MedrxivClusteringP2P",
    "MedrxivClusteringP2P.v2",
    "MedrxivClusteringS2S",
    "MedrxivClusteringS2S.v2",
    "TwentyNewsgroupsClustering",
    "TwentyNewsgroupsClustering.v2",
    "StackExchangeClustering",
    "StackExchangeClustering.v2",
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2P.v2",
    "RedditClustering",
    "RedditClustering.v2",
    "RedditClusteringP2P",
    "RedditClusteringP2P.v2",
    "STSBenchmark",
    "STSBenchmarkMultilingualSTS",  # translated, not trained on
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",
    "MrTidyRetrieval",
}


class _NVEmbedWrapper(InstructSentenceTransformerModel):
    """Inherited, because nvembed requires `sbert==2`, but it doesn't have tokenizers kwargs"""

    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str
        | Callable[[str, PromptType | None], str]
        | None = None,
        max_seq_length: int | None = None,
        apply_instruction_to_passages: bool = True,
        padding_side: str | None = None,
        add_eos_token: bool = False,
        prompts_dict: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        from sentence_transformers import __version__ as sbert_version

        required_transformers_version = "4.42.4"
        required_sbert_version = "2.7.0"

        if Version(transformers_version) != Version(required_transformers_version):
            raise RuntimeError(
                f"transformers version {transformers_version} is not match with required "
                f"install version {required_transformers_version} to run `nvidia/NV-Embed-v2`"
            )

        if Version(sbert_version) != Version(required_sbert_version):
            raise RuntimeError(
                f"sbert version {sbert_version} is not match with required "
                f"install version {required_sbert_version} to run `nvidia/NV-Embed-v2`"
            )

        requires_package(
            self, "flash_attn", model_name, "pip install 'mteb[flash_attention]'"
        )

        from sentence_transformers import SentenceTransformer

        if (
            isinstance(instruction_template, str)
            and "{instruction}" not in instruction_template
        ):
            raise ValueError(
                "Instruction template must contain the string '{instruction}'."
            )
        if instruction_template is None:
            logger.warning(
                "No instruction template provided. Instructions will be used as-is."
            )

        self.instruction_template = instruction_template

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, revision=revision, **kwargs)
        self.model.tokenizer.padding_side = padding_side
        self.model.tokenizer.add_eos_token = add_eos_token

        if max_seq_length:
            # https://github.com/huggingface/sentence-transformers/issues/3575
            self.model.max_seq_length = max_seq_length
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.prompts_dict = prompts_dict


NV_embed_v2 = ModelMeta(
    loader=_NVEmbedWrapper,
    loader_kwargs=dict(
        instruction_template=_instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="7604d305b621f14095a1aa23d351674c2859553a",
    release_date="2024-09-09",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=14975,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
    citation=NV_RETRIEVER_CITATION,
)

NV_embed_v1 = ModelMeta(
    loader=_NVEmbedWrapper,
    loader_kwargs=dict(
        instruction_template=_instruction_template,
        trust_remote_code=True,
        max_seq_length=32768,
        padding_side="right",
        # for nv-embed, we add eos token to each input example
        add_eos_token=True,
    ),
    name="nvidia/NV-Embed-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="570834afd5fef5bf3a3c2311a2b6e0a66f6f4f2c",
    release_date="2024-09-13",  # initial commit of hf model.
    n_parameters=7_850_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=14975,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/NV-Embed-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    training_datasets=nvidia_training_datasets,
    public_training_code=None,
    public_training_data=None,
    citation=NV_RETRIEVER_CITATION,
)

llama_embed_nemotron_evaluated_languages = [
    "afr-Latn",
    "amh-Ethi",
    "ara-Arab",
    "arq-Arab",
    "ary-Arab",
    "bbc-Latn",
    "ben-Beng",
    "bug-Latn",
    "bul-Cyrl",
    "cat-Latn",
    "ces-Latn",
    "cmn-Hans",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "est-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "hau-Latn",
    "hin-Deva",
    "hun-Latn",
    "hye-Armn",
    "ibo-Latn",
    "ind-Latn",
    "ita-Latn",
    "jav-Latn",
    "jpn-Jpan",
    "kin-Latn",
    "kor-Hang",
    "kor-Kore",
    "lav-Latn",
    "lin-Latn",
    "lug-Latn",
    "mad-Latn",
    "min-Latn",
    "mlt-Latn",
    "nld-Latn",
    "nob-Latn",
    "nor-Latn",
    "orm-Ethi",
    "pcm-Latn",
    "pol-Latn",
    "por-Latn",
    "ron-Latn",
    "run-Latn",
    "rus-Cyrl",
    "slk-Latn",
    "slv-Latn",
    "sna-Latn",
    "som-Latn",
    "spa-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "sun-Latn",
    "swa-Latn",
    "swe-Latn",
    "tel-Telu",
    "tha-Thai",
    "tir-Ethi",
    "tur-Latn",
    "vie-Latn",
    "xho-Latn",
    "yor-Latn",
    "zho-Hans",
]

TASK_PROMPTS = {
    # Classification
    "BulgarianStoreReviewSentimentClassfication": "Classify user reviews into positive or negative sentiment",
    "CzechProductReviewSentimentClassification": "Classify product reviews into positive or negative sentiment",
    "GreekLegalCodeClassification": "Given a greek legal text, classify its topic",
    "DBpediaClassification": "Given a Wikipedia articles, categorized it into classes based on its DBpedia ontology",
    "FinancialPhrasebankClassification": "Given financial news, categorized by sentiment into positive, negative, or neutral",
    "PoemSentimentClassification": "Given a poem, categorized by sentiment into positive, no_impact, negative or mixed",
    "TweetTopicSingleClassification": "Given a twitter, classify its topic",
    "EstonianValenceClassification": "Given a news article, categorized by sentiment into negatiivne, positiivne, neutraalne or vastuolulin",
    "FilipinoShopeeReviewsClassification": "Given a shop review, classify its rating on a scale from 1 to 5",
    "GujaratiNewsClassification": "Given a Gujarati news articles, classify its topic",
    "SentimentAnalysisHindi": "Given a hindi text, categorized by sentiment into positive, negative or neutral",
    "IndonesianIdClickbaitClassification": "Given an Indonesian news headlines, classify its into clickbait or non-clickbait",
    "ItaCaseholdClassification": "Given a judgments, classify its topic",
    "KorSarcasmClassification": "Given a twitter, categorized it into sarcasm or not_sarcasm",
    "KurdishSentimentClassification": "Given a text, categorized by sentiment into positive or negative",
    "MacedonianTweetSentimentClassification": "Given a Macedonian tweet, categorized by sentiment into positive, negative, or neutral",
    "AfriSentiClassification": "Given a text, categorized by sentiment into positive, negative, or neutral",
    "CataloniaTweetClassification": "Given a tweet, categorized by sentiment into AGAINST, FAVOR or NEUTRAL",
    "CyrillicTurkicLangClassification": "Given a text, classify its language",
    "IndicLangClassification": "Given a text, classify its language",
    "MasakhaNEWSClassification": "Classify the News in the given texts into one of the seven category: politics,sports,health,business,entertainment,technology,religion ",
    "MultiHateClassification": "Given a text, categorized by sentiment into hate or non-hate",
    "NusaParagraphEmotionClassification": "Given a paragraph, classify its emotion",
    "NusaX-senti": "Given a text, categorized by sentiment into positive or negative",
    "SwissJudgementClassification": "Given a news article, categorized it into approval or dismissal",
    "NepaliNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "OdiaNewsClassification": "Given a news article, categorized it into business, entertainment or sports",
    "PunjabiNewsClassification": "Given a news article, categorized it into two-classes",
    "PolEmo2.0-OUT": "Classify the sentiment of out-of-domain (products and school) online reviews",
    "PAC": 'Classify the sentence into one of the two types: "BEZPIECZNE_POSTANOWIENIE_UMOWNE" and "KLAUZULA_ABUZYWNA"',
    "SinhalaNewsClassification": "Given a news article, categorized it into political, business, technology, sports and Entertainment",
    "CSFDSKMovieReviewSentimentClassification": "Given a movie review, classify its rating on a scale from 0 to 5",
    "SiswatiNewsClassification": "Given a news article, classify its topic",
    "SlovakMovieReviewSentimentClassification": "Given a movie review, categorized it into positive or negative",
    "SwahiliNewsClassification": "Given a news article, classify its domain",
    "TswanaNewsClassification": "Given a news article, classify its topic",
    "IsiZuluNewsClassification": "Given a news article, classify its topic",
    # Clustering
    "WikiCitiesClustering": "Identify of Wikipedia articles of cities by country",
    "MasakhaNEWSClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "RomaniBibleClustering": "Identify verses from the Bible in Kalderash Romani by book.",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BigPatentClustering.v2": "Identify the category of documents from the Big Patent dataset",
    "AlloProfClusteringS2S.v2": "Identify the topic of document titles from Allo Prof dataset",
    "HALClusteringS2S.v2": "Identify the topic of titles from HAL",
    "SIB200ClusteringS2S": "Identify the category of documents",
    "WikiClusteringP2P.v2": "Identify the category of wiki passages",
    "PlscClusteringP2P.v2": "Identify the category of titles+abstracts from Library of Science",
    # Multilabel Classification
    "KorHateSpeechMLClassification": "Given a Korean online news comments, classify its fine-grained hate speech classes",
    "MalteseNewsClassification": "Given a maltese new, classify its topic",
    "MultiEURLEXMultilabelClassification": "Given a text, classify its topic",
    "BrazilianToxicTweetsClassification": "Given a tweet, classify its topic",
    # Retrieval
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "AILAStatutes": "Identifying the most relevant statutes for a given situation",
    "ArguAna": {
        "query": "Given a claim, find documents that refute the claim",
        "document": "Given a claim, find documents that refute the claim",
    },
    "HagridRetrieval": "Given a question, retrieve passages that answer the question",
    "LegalBenchCorporateLobbying": "Given a question, retrieve passages that answer the question",
    "LEMBPasskeyRetrieval": "Given a question, retrieve passages that answer the question",
    "BelebeleRetrieval": "Given a question, retrieve passages that answer the question",
    "MLQARetrieval": "Given a question, retrieve passages that answer the question",
    "StatcanDialogueDatasetRetrieval": "Given a question, retrieve passages that answer the question",
    "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question",
    "Core17InstructionRetrieval": "Given a question, retrieve passages that answer the question",
    "News21InstructionRetrieval": "Given a question, retrieve passages that answer the question",
    "Robust04InstructionRetrieval": "Given a question, retrieve passages that answer the question",
    "MIRACLRetrievalHardNegatives": "Given a question, retrieve passages that answer the question",
    # Reranking
    "WebLINXCandidatesReranking": "Given a question, retrieve passages that answer the question",
    "AlloprofReranking": "Given a question, retrieve passages that answer the question",
    "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question",
}

llama_embed_nemotron_training_datasets = {
    "AmazonReviewsClassification",
    "BioASQ",
    "EmotionClassification",
    "FEVER",
    "GooAQ",
    "HotpotQA",
    "MIRACLRetrieval",
    "MSMARCO",
    "MrTidyRetrieval",
    "Nemotron-CC-v2",
    "NFCorpus",
    "NQ",
    "PAQ",
    "RedditClustering.v2",
    "RedditClusteringP2P.v2",
    "SQuAD",
    "TriviaQA",
}


class LlamaEmbedNemotron(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
    ) -> None:
        required_transformers_version = "4.51.0"
        if Version(transformers_version) != Version(required_transformers_version):
            raise ImportError(
                f"{model_name} requires transformers library version {required_transformers_version}, but it was not found in your environment. "
                + f"If you want to load {model_name} model, please run `pip install 'mteb[llama-embed-nemotron]'` to install the required package."
            )

        requires_package(
            self, "flash_attn", model_name, "pip install 'mteb[flash_attention]'"
        )

        self.model_name = model_name
        self.revision = revision
        self.max_seq_length = 4096
        self.attn_implementation = (
            "flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_prompts = TASK_PROMPTS
        self.instruction_template = self._instruction_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
            revision=self.revision,
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation=self.attn_implementation,
            revision=self.revision,
        ).eval()

        self.model = self.model.to(self.device)

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
        """Encode sentences with task-specific instructions."""

        instruction = self._get_task_specific_instruction(task_metadata, prompt_type)

        prefix = self.format_instruction(instruction, prompt_type)
        return self._extract_embeddings(inputs, instruction=prefix, **kwargs)

    def _get_task_specific_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str:
        """Get instruction for a specific task, applying task-specific overrides."""
        # First, get the base instruction using custom prompts or task metadata
        instruction = self._get_base_instruction(task_metadata, prompt_type)

        # Apply task-type-specific overrides
        instruction = self._apply_task_type_overrides(
            instruction, task_metadata.type, task_metadata.name, prompt_type
        )

        return instruction

    def _get_base_instruction(
        self, task_metadata: TaskMetadata, prompt_type: PromptType | None
    ) -> str:
        """Get the base instruction from task-specific prompts or task metadata."""
        task_name = task_metadata.name

        # Check if task has custom prompt in TASK_PROMPTS
        if task_name in self.task_prompts:
            instruction = self.task_prompts[task_name]
            # Handle dict-based prompts (e.g., ArguAna with different query/document prompts)
            if isinstance(instruction, dict) and prompt_type:
                return instruction.get(prompt_type.value, "")
            return instruction if isinstance(instruction, str) else ""

        # Fall back to AbsEncoder's get_instruction method for task metadata
        return self.get_instruction(task_metadata, prompt_type)

    def _apply_task_type_overrides(
        self,
        instruction: str,
        task_type: str,
        task_name: str,
        prompt_type: PromptType | None,
    ) -> str:
        """Apply task-type-specific instruction overrides."""

        # For retrieval/reranking tasks, skip instruction for documents
        # unless it's a symmetric task (where both query and document use prompts)
        is_symmetric_task = task_name in self.task_prompts and isinstance(
            self.task_prompts[task_name], dict
        )

        if (
            ("Retrieval" in task_type or "Reranking" in task_type)
            and not is_symmetric_task
            and prompt_type == PromptType.document
        ):
            return ""

        # Override for STS and PairClassification tasks
        if task_type in ["STS", "PairClassification"]:
            return "Retrieve semantically similar text"

        # Override for BitextMining tasks
        if task_type in ["BitextMining"]:
            return "Retrieve parallel sentences"

        return instruction

    @staticmethod
    def _instruction_template(
        instruction: str, prompt_type: PromptType | None = None
    ) -> str:
        """Format instruction with the model-specific template."""

        return f"Instruct: {instruction}\nQuery: " if instruction else ""

    @staticmethod
    def average_pooling(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _extract_embeddings(
        self,
        dataloader: DataLoader[BatchedInput],
        instruction: str,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        all_embeddings = []
        for batch in tqdm(
            dataloader, desc="Extracting embeddings...", disable=not show_progress_bar
        ):
            with torch.inference_mode():
                inputs = self.tokenizer(
                    [instruction + t for t in batch["text"]],
                    max_length=self.max_seq_length,
                    truncation="longest_first",
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(**inputs)
                if outputs.last_hidden_state.dtype == torch.float16:
                    outputs.last_hidden_state = outputs.last_hidden_state.to(
                        torch.float32
                    )

                embeddings = self.average_pooling(
                    last_hidden_states=outputs.last_hidden_state,
                    attention_mask=inputs["attention_mask"],
                )

                embeddings = F.normalize(embeddings, dim=-1)

            all_embeddings.append(embeddings.contiguous())

        result = (
            torch.cat(all_embeddings, dim=0)
            .detach()
            .cpu()
            .numpy()
            .astype("float32")[: len(dataloader.dataset)]
        )

        del all_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


llama_embed_nemotron_8b = ModelMeta(
    loader=LlamaEmbedNemotron,
    name="nvidia/llama-embed-nemotron-8b",
    model_type=["dense"],
    languages=llama_embed_nemotron_evaluated_languages,
    open_weights=True,
    revision="84a375593d27d3528beb4e104822515659e093b4",
    release_date="2025-10-23",
    n_parameters=7_504_924_672,
    n_embedding_parameters=None,
    memory_usage_mb=28629,
    embed_dim=4096,
    license="https://huggingface.co/nvidia/llama-embed-nemotron-8b/blob/main/LICENSE",
    max_tokens=32768,
    reference="https://huggingface.co/nvidia/llama-embed-nemotron-8b",
    similarity_fn_name="cosine",
    framework=["PyTorch", "Sentence Transformers", "safetensors", "Transformers"],
    use_instructions=True,
    training_datasets=llama_embed_nemotron_training_datasets,
    public_training_code="https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/biencoder/llama_embed_nemotron_8b",
    public_training_data="https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1",
    contacts=["ybabakhin"],
    citation=LlamaEmbedNemotron_CITATION,
)


def _nemotron_rerank_model(model: str, revision: str, **kwargs) -> CrossEncoderWrapper:
    required_transformers_version = "4.47.1"

    if Version(transformers_version) != Version(required_transformers_version):
        raise RuntimeError(
            f"transformers version {transformers_version} is not match with required "
            f"install version {required_transformers_version} to run `nvidia/llama-nemotron-rerank-1b-v2`"
        )

    return CrossEncoderWrapper(
        model=model,
        revision=revision,
        **kwargs,
    )


nemotron_rerank_1b_v2 = ModelMeta(
    loader=_nemotron_rerank_model,
    loader_kwargs=dict(
        trust_remote_code=True,
        query_prefix="question:",
        passage_prefix=" \n \n passage:",
        model_kwargs={"torch_dtype": torch.float32},
    ),
    name="nvidia/llama-nemotron-rerank-1b-v2",
    revision="78efcfdc23b53a753f6c73f2d78b18132a34ac4d",
    release_date="2025-10-16",
    languages=["eng-Latn"],
    n_parameters=1235816448,
    memory_usage_mb=2357.0,
    max_tokens=4096,
    embed_dim=2048,
    license="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=set(
        # private
    ),
    adapted_from="meta-llama/Llama-3.2-1B",
    superseded_by=None,
    modalities=["text"],
    model_type=["cross-encoder"],
    citation=None,
    contacts=None,
)
