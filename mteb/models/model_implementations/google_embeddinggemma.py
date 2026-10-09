from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from mteb.models import SentenceTransformerEncoderWrapper
from mteb.models.model_implementations.google_gemini import GECKO_TRAINING_DATA
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput

logger = logging.getLogger(__name__)


MULTILINGUAL_EVALUATED_LANGUAGES = [
    "arb-Arab",
    "ben-Beng",
    "eng-Latn",
    "spa-Latn",
    "deu-Latn",
    "pes-Arab",
    "fin-Latn",
    "fra-Latn",
    "hin-Deva",
    "ind-Latn",
    "jpn-Jpan",
    "kor-Hang",
    "rus-Cyrl",
    "swh-Latn",
    "tel-Telu",
    "tha-Thai",
    "yor-Latn",
    "zho-Hant",
    "zho-Hans",
]


EMBEDDING_GEMMA_CITATION = """
@misc{vera2025embeddinggemmapowerfullightweighttext,
      title={EmbeddingGemma: Powerful and Lightweight Text Representations},
      author={Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and Daniel Cer and Alice Lisak and Min Choi and Lucas Gonzalez and Omar Sanseviero and Glenn Cameron and Ian Ballantyne and Kat Black and Kaifeng Chen and Weiyi Wang and Zhe Li and Gus Martins and Jinhyuk Lee and Mark Sherwood and Juyeong Ji and Renjie Wu and Jingxiao Zheng and Jyotinder Singh and Abheesht Sharma and Divyashree Sreepathihalli and Aashi Jain and Adham Elarabawy and AJ Co and Andreas Doumanoglou and Babak Samari and Ben Hora and Brian Potetz and Dahun Kim and Enrique Alfonseca and Fedor Moiseev and Feng Han and Frank Palma Gomez and Gustavo Hernández Ábrego and Hesen Zhang and Hui Hui and Jay Han and Karan Gill and Ke Chen and Koert Chen and Madhuri Shanbhogue and Michael Boratko and Paul Suganthan and Sai Meher Karthik Duddu and Sandeep Mariserla and Setareh Ariafar and Shanfeng Zhang and Shijie Zhang and Simon Baumgartner and Sonam Goenka and Steve Qiu and Tanmaya Dabral and Trevor Walker and Vikram Rao and Waleed Khawaja and Wenlei Zhou and Xiaoqi Ren and Ye Xia and Yichang Chen and Yi-Ting Chen and Zhe Dong and Zhongli Ding and Francesco Visin and Gaël Liu and Jiageng Zhang and Kathleen Kenealy and Michelle Casbon and Ravin Kumar and Thomas Mesnard and Zach Gleicher and Cormac Brick and Olivier Lacombe and Adam Roberts and Qin Yin and Yunhsuan Sung and Raphael Hoffmann and Tris Warkentin and Armand Joulin and Tom Duerig and Mojtaba Seyedhosseini},
      year={2025},
      eprint={2509.20354},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.20354},
}"""
EMBEDDING_GEMMA_2_PLACEHOLDER_CITATION = EMBEDDING_GEMMA_CITATION


embedding_gemma_300m = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-300m",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    release_date="2025-09-04",
    n_parameters=307_581_696,
    n_embedding_parameters=201_326_592,
    embed_dim=768,
    max_tokens=2048,
    license="gemma",
    reference="https://ai.google.dev/gemma/docs/embeddinggemma/model_card",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1155,
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)


# ---------------------------------------------------------------------------
# EmbeddingGemma 2 evaluation recipe
# ---------------------------------------------------------------------------
# The model-card numbers for MTEB(Multilingual, v2), MTEB(eng, v2), MTEB(Code, v1) and
# MIEB(lite) were produced with ONE instruction and ONE max sequence length PER TASK
# (the model was tuned against these), so a task-type prompt dictionary does not
# reproduce them. The exact per-task settings are listed below, grouped by benchmark
# because 17 tasks appear in both MTEB(Multilingual, v2) and MTEB(eng, v2) with
# different sequence lengths (and StackOverflowQA with a different instruction in
# MTEB(Code, v1)). ``EmbeddingGemma2Wrapper(recipe=...)`` selects the benchmark;
# without it the first benchmark containing the task wins, in the order below.
#
# Formatting (identical to the internal evaluation pipeline):
#   * query / symmetric input ........ "task: {instruction} | query: {text}"
#   * document side of asymmetric tasks "title: {title} | text: {text}"
#                                        (title = "none" when the corpus has no title)
#   * asymmetric = task type "Retrieval" + the MIEB retrieval tasks listed below;
#     Reranking tasks are symmetric.
#   * MIEB: max_seq_length 380 for every task (image tokens included); for the tasks in
#     _MIEB_NO_PROMPT_ON_IMAGE_INPUTS any input containing an image gets no prompt.
#   * bf16, mean pooling, L2-normalised, 768d (MRL truncation via ``embed_dim``, handled by
#     SentenceTransformer's ``truncate_dim``).
# Tasks outside these benchmarks fall back to a task-type instruction (same strings as
# the prompts in the model's config_sentence_transformers.json) and ``default_max_seq_length``.

_RECIPE_MTEB_MULTILINGUAL_V2: dict[str, tuple[str, int]] = {
    "AfriSentiClassification": ("classification", 512),
    "AILAStatutes": ("sentence similarity", 512),
    "AlloProfClusteringS2S.v2": ("code retrieval", 2048),
    "AlloprofReranking": ("search result", 2048),
    "AmazonCounterfactualClassification": ("classification", 512),
    "ArguAna": ("fact checking", 512),
    "ArmenianParaphrasePC": ("classification", 1024),
    "ArXivHierarchicalClusteringP2P": ("clustering", 512),
    "ArXivHierarchicalClusteringS2S": ("clustering", 512),
    "BelebeleRetrieval": ("question answering", 2048),
    "BibleNLPBitextMining": ("search result", 2048),
    "BigPatentClustering.v2": ("sentence similarity", 2048),
    "BiorxivClusteringP2P.v2": ("code retrieval", 2048),
    "BornholmBitextMining": ("sentence similarity", 512),
    "BrazilianToxicTweetsClassification": ("classification", 1024),
    "BUCC.v2": ("sentence similarity", 512),
    "BulgarianStoreReviewSentimentClassfication": ("classification", 512),
    "CataloniaTweetClassification": ("code retrieval", 2048),
    "CEDRClassification": ("classification", 512),
    "CLSClusteringP2P.v2": ("code retrieval", 2048),
    "Core17InstructionRetrieval": ("question answering", 2048),
    "CovidRetrieval": ("search result", 1024),
    "CSFDSKMovieReviewSentimentClassification": ("classification", 2048),
    "CTKFactsNLI": ("classification", 512),
    "CyrillicTurkicLangClassification": ("clustering", 1024),
    "CzechProductReviewSentimentClassification": ("classification", 1024),
    "DalajClassification": ("classification", 512),
    "DBpediaClassification": ("classification", 1024),
    "DiaBlaBitextMining": ("sentence similarity", 512),
    "EstonianValenceClassification": ("classification", 512),
    "FaroeseSTS": ("sentence similarity", 512),
    "FilipinoShopeeReviewsClassification": ("classification", 512),
    "FinancialPhrasebankClassification": ("sentence similarity", 512),
    "FinParaSTS": ("code retrieval", 2048),
    "FloresBitextMining": ("search result", 1024),
    "GermanSTSBenchmark": ("sentence similarity", 512),
    "GreekLegalCodeClassification": ("question answering", 2048),
    "GujaratiNewsClassification": ("clustering", 512),
    "HagridRetrieval": ("sentence similarity", 512),
    "HALClusteringS2S.v2": ("clustering", 512),
    "IN22GenBitextMining": ("sentence similarity", 512),
    "IndicCrosslingualSTS": ("search result", 1024),
    "IndicGenBenchFloresBitextMining": ("sentence similarity", 1024),
    "IndicLangClassification": ("clustering", 1024),
    "IndonesianIdClickbaitClassification": ("classification", 512),
    "indonli": ("sentence similarity", 1024),
    "IsiZuluNewsClassification": ("fact checking", 512),
    "ItaCaseholdClassification": ("clustering", 512),
    "JSICK": ("sentence similarity", 1024),
    "KorHateSpeechMLClassification": ("search result", 512),
    "KorSarcasmClassification": ("question answering", 512),
    "KurdishSentimentClassification": ("classification", 1024),
    "LegalBenchCorporateLobbying": ("search result", 2048),
    "LEMBPasskeyRetrieval": ("sentence similarity", 2048),
    "MacedonianTweetSentimentClassification": ("classification", 512),
    "MalteseNewsClassification": ("clustering", 2048),
    "MasakhaNEWSClassification": ("clustering", 1024),
    "MasakhaNEWSClusteringS2S": ("clustering", 512),
    "MassiveIntentClassification": ("classification", 1024),
    "MedrxivClusteringP2P.v2": ("clustering", 512),
    "MIRACLRetrievalHardNegatives": ("fact checking", 2048),
    "MLQARetrieval": ("search result", 2048),
    "MultiEURLEXMultilabelClassification": ("question answering", 1024),
    "MultiHateClassification": ("sentence similarity", 512),
    "NepaliNewsClassification": ("clustering", 512),
    "News21InstructionRetrieval": ("classification", 512),
    "NollySentiBitextMining": ("question answering", 1024),
    "NordicLangClassification": ("clustering", 2048),
    "NorwegianCourtsBitextMining": ("sentence similarity", 512),
    "NTREXBitextMining": ("search result", 2048),
    "NusaParagraphEmotionClassification": ("classification", 512),
    "NusaTranslationBitextMining": ("sentence similarity", 512),
    "NusaX-senti": ("classification", 512),
    "NusaXBitextMining": ("search result", 512),
    "OdiaNewsClassification": ("clustering", 1024),
    "OpusparcusPC": ("sentence similarity", 1024),
    "PAC": ("classification", 512),
    "PawsXPairClassification": ("question answering", 1024),
    "PlscClusteringP2P.v2": ("clustering", 512),
    "PoemSentimentClassification": ("sentence similarity", 512),
    "PolEmo2.0-OUT": ("classification", 512),
    "PpcPC": ("sentence similarity", 1024),
    "PunjabiNewsClassification": ("sentence similarity", 2048),
    "Robust04InstructionRetrieval": ("question answering", 1024),
    "RomaniBibleClustering": ("fact checking", 2048),
    "RTE3": ("sentence similarity", 512),
    "RuBQReranking": ("fact checking", 512),
    "ScalaClassification": ("sentence similarity", 2048),
    "SCIDOCS": ("question answering", 1024),
    "SemRel24STS": ("sentence similarity", 1024),
    "SentimentAnalysisHindi": ("classification", 512),
    "SIB200ClusteringS2S": ("sentence similarity", 512),
    "SICK-R": ("sentence similarity", 512),
    "SinhalaNewsClassification": ("clustering", 512),
    "SiswatiNewsClassification": ("search result", 1024),
    "SlovakMovieReviewSentimentClassification": ("classification", 512),
    "SpartQA": ("question answering", 512),
    "SprintDuplicateQuestions": ("search result", 512),
    "StackExchangeClustering.v2": ("clustering", 512),
    "StackOverflowQA": ("search result", 2048),
    "StatcanDialogueDatasetRetrieval": ("search result", 2048),
    "STS12": ("sentence similarity", 512),
    "STS13": ("sentence similarity", 512),
    "STS14": ("sentence similarity", 512),
    "STS15": ("sentence similarity", 1024),
    "STS17": ("sentence similarity", 512),
    "STS22.v2": ("sentence similarity", 1024),
    "STSB": ("sentence similarity", 512),
    "STSBenchmark": ("sentence similarity", 1024),
    "STSES": ("sentence similarity", 512),
    "SwahiliNewsClassification": ("clustering", 1024),
    "SwednClusteringP2P": ("clustering", 1024),
    "SwissJudgementClassification": ("question answering", 2048),
    "T2Reranking": ("fact checking", 512),
    "Tatoeba": ("sentence similarity", 1024),
    "TempReasonL1": ("search result", 512),
    "TERRa": ("sentence similarity", 512),
    "ToxicConversationsClassification": ("classification", 512),
    "TRECCOVID": ("search result", 2048),
    "TswanaNewsClassification": ("search result", 2048),
    "TweetTopicSingleClassification": ("clustering", 512),
    "TwitterHjerneRetrieval": ("search result", 2048),
    "TwitterURLCorpus": ("sentence similarity", 512),
    "VoyageMMarcoReranking": ("fact checking", 2048),
    "WebLINXCandidatesReranking": ("classification", 2048),
    "WikiCitiesClustering": ("clustering", 1024),
    "WikiClusteringP2P.v2": ("classification", 1024),
    "WikipediaRerankingMultilingual": ("sentence similarity", 1024),
    "WikipediaRetrievalMultilingual": ("search result", 512),
    "WinoGrande": ("search result", 512),
    "XNLI": ("sentence similarity", 512),
}

_RECIPE_MTEB_ENG_V2: dict[str, tuple[str, int]] = {
    "AmazonCounterfactualClassification": ("classification", 512),
    "ArguAna": ("fact checking", 2048),
    "ArXivHierarchicalClusteringP2P": ("clustering", 2048),
    "ArXivHierarchicalClusteringS2S": ("clustering", 512),
    "AskUbuntuDupQuestions": ("search result", 2048),
    "Banking77Classification": ("classification", 512),
    "BiorxivClusteringP2P.v2": ("code retrieval", 1024),
    "BIOSSES": ("search result", 2048),
    "ClimateFEVERHardNegatives": ("question answering", 1024),
    "CQADupstackGamingRetrieval": ("search result", 2048),
    "CQADupstackUnixRetrieval": ("question answering", 512),
    "FEVERHardNegatives": ("search result", 1024),
    "FiQA2018": ("question answering", 2048),
    "HotpotQAHardNegatives": ("question answering", 1024),
    "ImdbClassification": ("classification", 2048),
    "MassiveIntentClassification": ("classification", 512),
    "MassiveScenarioClassification": ("classification", 2048),
    "MedrxivClusteringP2P.v2": ("clustering", 2048),
    "MedrxivClusteringS2S.v2": ("clustering", 2048),
    "MindSmallReranking": ("clustering", 512),
    "MTOPDomainClassification": ("classification", 2048),
    "SCIDOCS": ("question answering", 2048),
    "SICK-R": ("sentence similarity", 2048),
    "SprintDuplicateQuestions": ("search result", 2048),
    "StackExchangeClustering.v2": ("clustering", 2048),
    "StackExchangeClusteringP2P.v2": ("clustering", 512),
    "STS12": ("sentence similarity", 512),
    "STS13": ("sentence similarity", 1024),
    "STS14": ("sentence similarity", 2048),
    "STS15": ("sentence similarity", 2048),
    "STS17": ("sentence similarity", 512),
    "STS22.v2": ("sentence similarity", 512),
    "STSBenchmark": ("sentence similarity", 2048),
    "SummEvalSummarization.v2": ("sentence similarity", 2048),
    "Touche2020Retrieval.v3": ("question answering", 1024),
    "ToxicConversationsClassification": ("classification", 512),
    "TRECCOVID": ("search result", 1024),
    "TweetSentimentExtractionClassification": ("classification", 2048),
    "TwentyNewsgroupsClustering.v2": ("clustering", 2048),
    "TwitterSemEval2015": ("sentence similarity", 512),
    "TwitterURLCorpus": ("sentence similarity", 1024),
}

_RECIPE_MTEB_CODE_V1: dict[str, tuple[str, int]] = {
    "AppsRetrieval": ("code retrieval", 2048),
    "CodeEditSearchRetrieval": ("code retrieval", 512),
    "CodeFeedbackMT": ("code retrieval", 2048),
    "CodeFeedbackST": ("code retrieval", 2048),
    "CodeSearchNetCCRetrieval": ("code retrieval", 1024),
    "CodeSearchNetRetrieval": ("code retrieval", 2048),
    "CodeTransOceanContest": ("code retrieval", 2048),
    "CodeTransOceanDL": ("code retrieval", 1024),
    "COIRCodeSearchNetRetrieval": ("code retrieval", 2048),
    "CosQA": ("code retrieval", 1024),
    "StackOverflowQA": ("code retrieval", 2048),
    "SyntheticText2SQL": ("code retrieval", 2048),
}

_RECIPE_MIEB_LITE: dict[str, tuple[str, int]] = {
    "AROCocoOrder": ("question answering", 380),
    "AROFlickrOrder": ("clustering", 380),
    "AROVisualAttribution": ("search result", 380),
    "AROVisualRelation": ("classification", 380),
    "BLINKIT2IMultiChoice": ("sentence similarity", 380),
    "CIFAR100ZeroShot": ("sentence similarity", 380),
    "CIRRIT2IRetrieval": ("clustering", 380),
    "Country211": ("clustering", 380),
    "Country211ZeroShot": ("code retrieval", 380),
    "CUB200I2IRetrieval": ("clustering", 380),
    "CVBenchCount": ("code retrieval", 380),
    "CVBenchDepth": ("sentence similarity", 380),
    "CVBenchDistance": ("classification", 380),
    "CVBenchRelation": ("classification", 380),
    "DTD": ("question answering", 380),
    "EuroSAT": ("sentence similarity", 380),
    "Fashion200kI2TRetrieval": ("search result", 380),
    "FER2013ZeroShot": ("question answering", 380),
    "FGVCAircraftZeroShot": ("question answering", 380),
    "Food101ZeroShot": ("question answering", 380),
    "GTSRB": ("question answering", 380),
    "HatefulMemesI2TRetrieval": ("question answering", 380),
    "ImageCoDe": ("fact checking", 380),
    "ImageNetDog15Clustering": ("clustering", 380),
    "InfoSeekIT2TRetrieval": ("fact checking", 380),
    "NIGHTSI2IRetrieval": ("clustering", 380),
    "OVENIT2TRetrieval": ("question answering", 380),
    "OxfordPets": ("fact checking", 380),
    "OxfordPetsZeroShot": ("fact checking", 380),
    "PatchCamelyon": ("clustering", 380),
    "RESISC45": ("sentence similarity", 380),
    "RP2kI2IRetrieval": ("clustering", 380),
    "StanfordCarsZeroShot": ("question answering", 380),
    "STS13VisualSTS": ("sentence similarity", 380),
    "STS15VisualSTS": ("sentence similarity", 380),
    "STS17MultilingualVisualSTS": (
        "sentence similarity",
        380,
    ),  # sub-task name used by the aggregate task
    "STSBenchmarkMultilingualVisualSTS": (
        "sentence similarity",
        380,
    ),  # sub-task name used by the aggregate task
    "SUN397": ("sentence similarity", 380),
    "TinyImageNetClustering": ("fact checking", 380),
    "VidoreDocVQARetrieval": ("question answering", 380),
    "VidoreInfoVQARetrieval": ("search result", 380),
    "VidoreShiftProjectRetrieval": ("search result", 380),
    "VidoreSyntheticDocQAAIRetrieval": ("search result", 380),
    "VidoreTabfquadRetrieval": ("question answering", 380),
    "VidoreTatdqaRetrieval": ("search result", 380),
    "VisualNewsI2TRetrieval": ("fact checking", 380),
    "VisualSTS-b-Multilingual": ("sentence similarity", 380),
    "VisualSTS17Multilingual": ("sentence similarity", 380),
    "VQA2IT2TRetrieval": ("clustering", 380),
    "WebQAT2ITRetrieval": ("search result", 380),
    "Winoground": ("search result", 380),
    "WITT2IRetrieval": ("question answering", 380),
    "XM3600T2IRetrieval": ("question answering", 380),
}

EMBEDDING_GEMMA_2_RECIPES: dict[str, dict[str, tuple[str, int]]] = {
    "MTEB(Multilingual, v2)": _RECIPE_MTEB_MULTILINGUAL_V2,
    "MTEB(eng, v2)": _RECIPE_MTEB_ENG_V2,
    "MTEB(Code, v1)": _RECIPE_MTEB_CODE_V1,
    "MIEB(lite)": _RECIPE_MIEB_LITE,
}

# MIEB tasks encoded asymmetrically (query template vs. "title: ... | text: ..." template).
_MIEB_ASYMMETRIC_TASKS = {
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "Fashion200kI2TRetrieval",
    "HatefulMemesI2TRetrieval",
    "InfoSeekIT2TRetrieval",
    "NIGHTSI2IRetrieval",
    "OVENIT2TRetrieval",
    "RP2kI2IRetrieval",
    "VQA2IT2TRetrieval",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VisualNewsI2TRetrieval",
    "WITT2IRetrieval",
    "WebQAT2ITRetrieval",
    "XM3600T2IRetrieval",
}

# MIEB tasks where inputs that contain an image are encoded WITHOUT any prompt
# (image-only inputs -> image only; image+text inputs -> image + raw text).
_MIEB_NO_PROMPT_ON_IMAGE_INPUTS = {
    "AROFlickrOrder",
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "CVBenchCount",
    "Country211ZeroShot",
    "Food101ZeroShot",
    "ImageCoDe",
    "ImageNetDog15Clustering",
    "NIGHTSI2IRetrieval",
    "OxfordPetsZeroShot",
    "PatchCamelyon",
    "RP2kI2IRetrieval",
    "VQA2IT2TRetrieval",
    "VidoreTatdqaRetrieval",
    "VisualNewsI2TRetrieval",
    "WITT2IRetrieval",
    "Winoground",
    "XM3600T2IRetrieval",
}

# Fallback for tasks outside the four benchmarks.
_TASK_TYPE_INSTRUCTION = {
    "Retrieval": "search result",
    "Reranking": "search result",
    "InstructionRetrieval": "search result",
    "InstructionReranking": "search result",
    "BitextMining": "search result",
    "Classification": "classification",
    "MultilabelClassification": "classification",
    "Clustering": "clustering",
    "STS": "sentence similarity",
    "PairClassification": "sentence similarity",
    "Summarization": "sentence similarity",
}


class EmbeddingGemma2Wrapper(SentenceTransformerEncoderWrapper):
    """SentenceTransformer wrapper that applies the EmbeddingGemma 2 per-task recipe.

    Everything (prompt handling, multimodal inputs, MRL truncation via ``embed_dim``,
    normalisation) is inherited from ``SentenceTransformerEncoderWrapper``; this class only
    (1) loads just the requested modality towers, (2) sets the per-task ``max_seq_length``
    and (3) writes the per-task query / document templates into the batch text before
    delegating, because the document template ``title: {title} | text: {body}`` is not a
    plain prefix of mteb's ``"{title} {body}"`` corpus text.

    Args:
        model_name: HF model id.
        revision: HF revision.
        modalities: Modalities to load; towers for other modalities are dropped at load
            time (``vision_config=None`` / ``audio_config=None``). The model-card text
            numbers use ``("text",)`` and the MIEB numbers ``("text", "image")``.
        recipe: Which benchmark's per-task settings to use (a key of
            ``EMBEDDING_GEMMA_2_RECIPES``); ``None`` picks the first benchmark listing the task.
        default_max_seq_length: Sequence length for tasks not covered by the recipe.
        **kwargs: Forwarded to ``SentenceTransformerEncoderWrapper`` (e.g. ``embed_dim``).
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        revision: str | None = None,
        *,
        modalities: tuple[str, ...] = ("text",),
        recipe: str | None = None,
        default_max_seq_length: int = 512,
        **kwargs: Any,
    ) -> None:
        if recipe is not None and recipe not in EMBEDDING_GEMMA_2_RECIPES:
            raise ValueError(
                f"Unknown recipe {recipe!r}; expected one of {list(EMBEDDING_GEMMA_2_RECIPES)}"
            )
        config_kwargs = dict(kwargs.pop("config_kwargs", {}))
        if "image" not in modalities and "video" not in modalities:
            config_kwargs["vision_config"] = None
        if "audio" not in modalities:
            config_kwargs["audio_config"] = None
        model_kwargs = {"torch_dtype": torch.bfloat16, **kwargs.pop("model_kwargs", {})}
        super().__init__(
            model_name,
            revision=revision,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            **kwargs,
        )
        # Templates are written into the batch text in encode(). Disable both mteb's
        # model_prompts and the model's built-in "query"/"document" prompts, which
        # SentenceTransformer.encode_query/encode_document would otherwise prepend as well.
        self.model_prompts = {}
        self.model.prompts = {}
        self.model.default_prompt_name = None
        self.mteb_model_meta.modalities = list(modalities)
        self.recipe = recipe
        self.default_max_seq_length = default_max_seq_length

    def _task_settings(self, task_metadata: TaskMetadata) -> tuple[str, int]:
        """Return the recipe's (instruction, max_seq_length) for a task."""
        name = task_metadata.name
        tables = [EMBEDDING_GEMMA_2_RECIPES[self.recipe]] if self.recipe else []
        tables += list(EMBEDDING_GEMMA_2_RECIPES.values())
        for table in tables:
            if name in table:
                return table[name]
        logger.warning(
            "No EmbeddingGemma 2 recipe entry for %s; using task-type fallback.", name
        )
        instruction = _TASK_TYPE_INSTRUCTION.get(task_metadata.type, "search result")
        return instruction, self.default_max_seq_length

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
        name = task_metadata.name
        instruction, seq_len = self._task_settings(task_metadata)
        self.model.max_seq_length = seq_len
        asymmetric = task_metadata.type == "Retrieval" or name in _MIEB_ASYMMETRIC_TASKS
        prompt_on_images = name not in _MIEB_NO_PROMPT_ON_IMAGE_INPUTS
        is_document = asymmetric and prompt_type == PromptType.document
        query_prefix = f"task: {instruction} | query: "

        base_collate = inputs.collate_fn

        def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            batch = base_collate(rows)
            if "image" in batch and not prompt_on_images:
                return batch  # image inputs of this task get no prompt at all
            if is_document:
                bodies = batch.get("body", batch.get("text", []))
                titles = batch.get("title", [""] * len(bodies))
                batch["text"] = [
                    f"title: {t.strip() or 'none'} | text: {b}"
                    for t, b in zip(titles, bodies, strict=True)
                ]
            else:
                texts = batch["text"] if "text" in batch else [""] * len(batch["image"])
                batch["text"] = [query_prefix + t for t in texts]
            return batch

        inputs.collate_fn = collate
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )


def _embedding_gemma_2_loader(
    model_name: str,
    revision: str | None = None,
    *,
    modalities: tuple[str, ...] = ("text",),
    **kwargs: Any,
) -> EmbeddingGemma2Wrapper:
    return EmbeddingGemma2Wrapper(
        "google/embeddinggemma-2", revision=revision, modalities=modalities, **kwargs
    )


def _embedding_gemma_2_text_loader(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> EmbeddingGemma2Wrapper:
    return _embedding_gemma_2_loader(
        model_name, revision, modalities=("text",), **kwargs
    )


def _embedding_gemma_2_vision_text_loader(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> EmbeddingGemma2Wrapper:
    kwargs.setdefault("default_max_seq_length", 380)
    return _embedding_gemma_2_loader(
        model_name, revision, modalities=("text", "image", "video"), **kwargs
    )


def _embedding_gemma_2_audio_text_loader(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> EmbeddingGemma2Wrapper:
    return _embedding_gemma_2_loader(
        model_name, revision, modalities=("text", "audio"), **kwargs
    )


def _embedding_gemma_2_full_loader(
    model_name: str, revision: str | None = None, **kwargs: Any
) -> EmbeddingGemma2Wrapper:
    return _embedding_gemma_2_loader(
        model_name, revision, modalities=("text", "image", "audio", "video"), **kwargs
    )


embedding_gemma_2 = ModelMeta(
    loader=_embedding_gemma_2_full_loader,
    name="google/embeddinggemma-2",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=744_371_992,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=2840,
    modalities=["text", "image", "audio", "video"],
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)


embedding_gemma_2_text = ModelMeta(
    loader=_embedding_gemma_2_text_loader,
    name="google/embeddinggemma-2-text",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=271_002_624,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1034,
    modalities=["text"],
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)


embedding_gemma_2_vision_text = ModelMeta(
    loader=_embedding_gemma_2_vision_text_loader,
    name="google/embeddinggemma-2-vision-text",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=438_760_448,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=1674,
    modalities=["text", "image", "video"],
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)


embedding_gemma_2_audio_text = ModelMeta(
    loader=_embedding_gemma_2_audio_text_loader,
    name="google/embeddinggemma-2-audio-text",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    n_parameters=576_613_664,
    n_embedding_parameters=134_217_728,
    embed_dim=768,
    max_tokens=8192,
    license="gemma",
    reference="https://huggingface.co/google/embeddinggemma-2",
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=GECKO_TRAINING_DATA,
    similarity_fn_name="cosine",
    memory_usage_mb=2200,
    modalities=["text", "audio"],
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)
