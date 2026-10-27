from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.languages import PROGRAMMING_LANGS
from mteb.models.model_implementations.google_gemini import GECKO_TRAINING_DATA
from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import (
    SentenceTransformerEncoderWrapper,
    _setup_modality_collator,
)
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput


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
# The numbers for MTEB(Multilingual, v2), MTEB(eng, v2), MTEB(Code, v1), MIEB(lite)
# and MAEB were produced with one instruction per task, so a task-type prompt
# dictionary does not reproduce them. Instructions that did not match the task were
# replaced by the task-type instruction (_task_type_instruction); the submitted instruction is
# kept as a comment on those entries.
# Formatting:
#   * query / symmetric input ........ "task: {instruction} | query: {text}"
#   * document side of asymmetric tasks "title: {title} | text: {text}"
#                                        (title = "none" when the corpus has no title)
#   * asymmetric = retrieval tasks, based on metadata and on an explicit list for
#     MIEB below.
#   * Text-only: the model's default max_seq_length for every task.
#   * MIEB: max_seq_length 380 for every task (image tokens included); for the tasks in
#     _MIEB_NO_PROMPT_ON_IMAGE_INPUTS any input containing an image gets no prompt.
#   * MAEB: max_seq_length 1024 for every task (audio tokens included). Raw audio
#     inputs are resampled to 16 kHz float32 mono with channel-averaging prior to
#     tokenization. All audio documents and retrieval corpora are embedded with prefix
#     "title: none | text: <|audio|>".
#   * bf16, mean pooling, L2-normalised, 768d (MRL truncation via ``embed_dim``).
# Tasks outside these benchmarks fall back to a simplified task-type instruction.

EMBEDDING_GEMMA_2_TASK_INSTRUCTIONS: dict[str, str] = {
    "AfriSentiClassification": "classification",
    "AILAStatutes": "search result",  # originally submitted as "sentence similarity"
    "AlloProfClusteringS2S.v2": "clustering",  # originally submitted as "code retrieval"
    "AlloprofReranking": "search result",
    "AmazonCounterfactualClassification": "classification",
    "AppsRetrieval": "code retrieval",
    "ArguAna": "fact checking",
    "ArmenianParaphrasePC": "classification",
    "AROCocoOrder": "sentence similarity",  # originally submitted as "question answering"
    "AROFlickrOrder": "sentence similarity",  # originally submitted as "clustering"
    "AROVisualAttribution": "search result",
    "AROVisualRelation": "classification",
    "ArXivHierarchicalClusteringP2P": "clustering",
    "ArXivHierarchicalClusteringS2S": "clustering",
    "AskUbuntuDupQuestions": "search result",
    "Banking77Classification": "classification",
    "BeijingOpera": "classification",
    "BelebeleRetrieval": "question answering",
    "BibleNLPBitextMining": "search result",
    "BigPatentClustering.v2": "clustering",  # originally submitted as "sentence similarity"
    "BiorxivClusteringP2P.v2": "clustering",  # originally submitted as "code retrieval"
    "BIOSSES": "sentence similarity",  # originally submitted as "search result"
    "BirdCLEF": "classification",  # originally submitted as "sentence similarity"
    "BLINKIT2IMultiChoice": "search result",  # originally submitted as "sentence similarity"
    "BornholmBitextMining": "sentence similarity",
    "BrazilianToxicTweetsClassification": "classification",
    "BUCC.v2": "sentence similarity",
    "BulgarianStoreReviewSentimentClassfication": "classification",
    "CataloniaTweetClassification": "classification",  # originally submitted as "code retrieval"
    "CEDRClassification": "classification",
    "CIFAR100ZeroShot": "sentence similarity",
    "CIRRIT2IRetrieval": "search result",  # originally submitted as "clustering"
    "ClimateFEVERHardNegatives": "question answering",
    "ClothoT2ARetrieval": "search result",
    "CLSClusteringP2P.v2": "clustering",  # originally submitted as "code retrieval"
    "CodeEditSearchRetrieval": "code retrieval",
    "CodeFeedbackMT": "code retrieval",
    "CodeFeedbackST": "code retrieval",
    "CodeSearchNetCCRetrieval": "code retrieval",
    "CodeSearchNetRetrieval": "code retrieval",
    "CodeTransOceanContest": "code retrieval",
    "CodeTransOceanDL": "code retrieval",
    "COIRCodeSearchNetRetrieval": "code retrieval",
    "CommonLanguageAgeDetection": "clustering",
    "CommonVoiceMini21T2ARetrieval": "sentence similarity",
    "Core17InstructionRetrieval": "question answering",
    "CosQA": "code retrieval",
    "Country211": "clustering",
    "Country211ZeroShot": "classification",  # originally submitted as "code retrieval"
    "CovidRetrieval": "search result",
    "CQADupstackGamingRetrieval": "search result",
    "CQADupstackUnixRetrieval": "question answering",
    "CREMA_D": "clustering",
    "CREMA_DClustering": "clustering",  # originally submitted as "fact checking"
    "CREMADPairClassification": "clustering",
    "CSFDSKMovieReviewSentimentClassification": "classification",
    "CTKFactsNLI": "classification",
    "CUB200I2IRetrieval": "clustering",
    "CVBenchCount": "question answering",  # originally submitted as "code retrieval"
    "CVBenchDepth": "question answering",  # originally submitted as "sentence similarity"
    "CVBenchDistance": "classification",
    "CVBenchRelation": "classification",
    "CyrillicTurkicLangClassification": "clustering",
    "CzechProductReviewSentimentClassification": "classification",
    "DalajClassification": "classification",
    "DBpediaClassification": "classification",
    "DiaBlaBitextMining": "sentence similarity",
    "DTD": "classification",  # originally submitted as "question answering"
    "EstonianValenceClassification": "classification",
    "EuroSAT": "classification",  # originally submitted as "sentence similarity"
    "FaroeseSTS": "sentence similarity",
    "Fashion200kI2TRetrieval": "search result",
    "FER2013ZeroShot": "classification",  # originally submitted as "question answering"
    "FEVERHardNegatives": "search result",
    "FGVCAircraftZeroShot": "classification",  # originally submitted as "question answering"
    "FilipinoShopeeReviewsClassification": "classification",
    "FinancialPhrasebankClassification": "classification",  # originally submitted as "sentence similarity"
    "FinParaSTS": "sentence similarity",  # originally submitted as "code retrieval"
    "FiQA2018": "question answering",
    "FleursT2ARetrieval": "search result",
    "FloresBitextMining": "search result",
    "Food101ZeroShot": "classification",  # originally submitted as "question answering"
    "FSD2019Kaggle": "classification",
    "GermanSTSBenchmark": "sentence similarity",
    "GigaSpeechT2ARetrieval": "search result",  # originally submitted as "question answering"
    "GreekLegalCodeClassification": "classification",  # originally submitted as "question answering"
    "GTSRB": "classification",  # originally submitted as "question answering"
    "GTZANAudioReranking": "search result",  # originally submitted as "code retrieval"
    "GTZANGenre": "classification",  # originally submitted as "fact checking"
    "GujaratiNewsClassification": "clustering",
    "HagridRetrieval": "search result",  # originally submitted as "sentence similarity"
    "HALClusteringS2S.v2": "clustering",
    "HatefulMemesI2TRetrieval": "search result",  # originally submitted as "question answering"
    "HotpotQAHardNegatives": "question answering",
    "IEMOCAPGender": "clustering",
    "ImageCoDe": "sentence similarity",  # originally submitted as "fact checking"
    "ImageNetDog15Clustering": "clustering",
    "ImdbClassification": "classification",
    "IN22GenBitextMining": "sentence similarity",
    "IndicCrosslingualSTS": "sentence similarity",  # originally submitted as "search result"
    "IndicGenBenchFloresBitextMining": "sentence similarity",
    "IndicLangClassification": "clustering",
    "IndonesianIdClickbaitClassification": "classification",
    "indonli": "sentence similarity",
    "InfoSeekIT2TRetrieval": "search result",  # originally submitted as "fact checking"
    "IsiZuluNewsClassification": "classification",  # originally submitted as "fact checking"
    "ItaCaseholdClassification": "clustering",
    "JamAltArtistA2ARetrieval": "search result",  # originally submitted as "fact checking"
    "JamAltLyricA2TRetrieval": "search result",
    "JSICK": "sentence similarity",
    "KorHateSpeechMLClassification": "classification",  # originally submitted as "search result"
    "KorSarcasmClassification": "classification",  # originally submitted as "question answering"
    "KurdishSentimentClassification": "classification",
    "LegalBenchCorporateLobbying": "search result",
    "LEMBPasskeyRetrieval": "search result",  # originally submitted as "sentence similarity"
    "MacedonianTweetSentimentClassification": "classification",
    "MACST2ARetrieval": "search result",  # originally submitted as "sentence similarity"
    "MalteseNewsClassification": "clustering",
    "MasakhaNEWSClassification": "clustering",
    "MasakhaNEWSClusteringS2S": "clustering",
    "MassiveIntentClassification": "classification",
    "MassiveScenarioClassification": "classification",
    "MedrxivClusteringP2P.v2": "clustering",
    "MedrxivClusteringS2S.v2": "clustering",
    "MInDS14": "classification",  # originally submitted as "question answering"
    "MindSmallReranking": "search result",  # originally submitted as "clustering"
    "MIRACLRetrievalHardNegatives": "search result",  # originally submitted as "fact checking"
    "MLQARetrieval": "search result",
    "MridinghamTonic": "classification",  # originally submitted as "question answering"
    "MTOPDomainClassification": "classification",
    "MultiEURLEXMultilabelClassification": "classification",  # originally submitted as "question answering"
    "MultiHateClassification": "classification",  # originally submitted as "sentence similarity"
    "NepaliNewsClassification": "clustering",
    "News21InstructionRetrieval": "search result",  # originally submitted as "classification"
    "NIGHTSI2IRetrieval": "search result",  # originally submitted as "clustering"
    "NMSQAPairClassification": "sentence similarity",
    "NollySentiBitextMining": "sentence similarity",  # originally submitted as "question answering"
    "NordicLangClassification": "clustering",
    "NorwegianCourtsBitextMining": "sentence similarity",
    "NTREXBitextMining": "search result",
    "NusaParagraphEmotionClassification": "classification",
    "NusaTranslationBitextMining": "sentence similarity",
    "NusaX-senti": "classification",
    "NusaXBitextMining": "search result",
    "OdiaNewsClassification": "clustering",
    "OpusparcusPC": "sentence similarity",
    "OVENIT2TRetrieval": "question answering",
    "OxfordPets": "classification",  # originally submitted as "fact checking"
    "OxfordPetsZeroShot": "classification",  # originally submitted as "fact checking"
    "PAC": "classification",
    "PatchCamelyon": "clustering",
    "PawsXPairClassification": "sentence similarity",  # originally submitted as "question answering"
    "PlscClusteringP2P.v2": "clustering",
    "PoemSentimentClassification": "classification",  # originally submitted as "sentence similarity"
    "PolEmo2.0-OUT": "classification",
    "PpcPC": "sentence similarity",
    "PunjabiNewsClassification": "classification",  # originally submitted as "sentence similarity"
    "RavdessZeroshot": "clustering",
    "RESISC45": "classification",  # originally submitted as "sentence similarity"
    "Robust04InstructionRetrieval": "question answering",
    "RomaniBibleClustering": "clustering",  # originally submitted as "fact checking"
    "RP2kI2IRetrieval": "clustering",
    "RTE3": "sentence similarity",
    "RuBQReranking": "search result",  # originally submitted as "fact checking"
    "ScalaClassification": "classification",  # originally submitted as "sentence similarity"
    "SCIDOCS": "question answering",
    "SemRel24STS": "sentence similarity",
    "SentimentAnalysisHindi": "classification",
    "SIB200ClusteringS2S": "clustering",  # originally submitted as "sentence similarity"
    "SIBFLEURS": "clustering",
    "SICK-R": "sentence similarity",
    "SinhalaNewsClassification": "clustering",
    "SiswatiNewsClassification": "classification",  # originally submitted as "search result"
    "SlovakMovieReviewSentimentClassification": "classification",
    "SpartQA": "question answering",
    "SpeechCommandsZeroshotv0.02": "sentence similarity",
    "SpokenSQuADT2ARetrieval": "search result",  # originally submitted as "fact checking"
    "SprintDuplicateQuestions": "search result",
    "StackExchangeClustering.v2": "clustering",
    "StackExchangeClusteringP2P.v2": "clustering",
    "StackOverflowQA": "code retrieval",
    "StanfordCarsZeroShot": "classification",  # originally submitted as "question answering"
    "StatcanDialogueDatasetRetrieval": "search result",
    "STS12": "sentence similarity",
    "STS13": "sentence similarity",
    "STS13VisualSTS": "sentence similarity",
    "STS14": "sentence similarity",
    "STS15": "sentence similarity",
    "STS15VisualSTS": "sentence similarity",
    "STS17": "sentence similarity",
    "STS17MultilingualVisualSTS": "sentence similarity",
    "STS22.v2": "sentence similarity",
    "STSB": "sentence similarity",
    "STSBenchmark": "sentence similarity",
    "STSBenchmarkMultilingualVisualSTS": "sentence similarity",
    "STSES": "sentence similarity",
    "SummEvalSummarization.v2": "sentence similarity",
    "SUN397": "classification",  # originally submitted as "sentence similarity"
    "SwahiliNewsClassification": "clustering",
    "SwednClusteringP2P": "clustering",
    "SwissJudgementClassification": "classification",  # originally submitted as "question answering"
    "SyntheticText2SQL": "code retrieval",
    "T2Reranking": "question answering",  # originally submitted as "fact checking"
    "Tatoeba": "sentence similarity",
    "TempReasonL1": "search result",
    "TERRa": "sentence similarity",
    "TinyImageNetClustering": "clustering",  # originally submitted as "fact checking"
    "Touche2020Retrieval.v3": "question answering",
    "ToxicConversationsClassification": "classification",
    "TRECCOVID": "search result",
    "TswanaNewsClassification": "classification",  # originally submitted as "search result"
    "TweetSentimentExtractionClassification": "classification",
    "TweetTopicSingleClassification": "clustering",
    "TwentyNewsgroupsClustering.v2": "clustering",
    "TwitterHjerneRetrieval": "search result",
    "TwitterSemEval2015": "sentence similarity",
    "TwitterURLCorpus": "sentence similarity",
    "UrbanSound8KT2ARetrieval": "search result",  # originally submitted as "fact checking"
    "VehicleSoundClustering": "clustering",  # originally submitted as "sentence similarity"
    "VidoreDocVQARetrieval": "question answering",
    "VidoreInfoVQARetrieval": "search result",
    "VidoreShiftProjectRetrieval": "search result",
    "VidoreSyntheticDocQAAIRetrieval": "search result",
    "VidoreTabfquadRetrieval": "question answering",
    "VidoreTatdqaRetrieval": "search result",
    "VisualNewsI2TRetrieval": "search result",  # originally submitted as "fact checking"
    "VisualSTS-b-Multilingual": "sentence similarity",
    "VisualSTS17Multilingual": "sentence similarity",
    "VoxCelebSA": "classification",
    "VoxPopuliAccentPairClassification": "clustering",
    "VoxPopuliGenderClustering": "clustering",  # originally submitted as "sentence similarity"
    "VoxPopuliLanguageID": "clustering",
    "VoyageMMarcoReranking": "search result",  # originally submitted as "fact checking"
    "VQA2IT2TRetrieval": "search result",  # originally submitted as "clustering"
    "WebLINXCandidatesReranking": "code retrieval",  # originally submitted as "classification"
    "WebQAT2ITRetrieval": "search result",
    "WikiCitiesClustering": "clustering",
    "WikiClusteringP2P.v2": "classification",
    "WikipediaRerankingMultilingual": "search result",  # originally submitted as "sentence similarity"
    "WikipediaRetrievalMultilingual": "search result",
    "WinoGrande": "search result",
    "Winoground": "search result",
    "WITT2IRetrieval": "search result",  # originally submitted as "question answering"
    "XM3600T2IRetrieval": "search result",  # originally submitted as "question answering"
    "XNLI": "sentence similarity",
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

# Instruction fallback for tasks outside the evaluated benchmarks.
_TASK_TYPE_INSTRUCTION = {
    "retrieval": "search result",
    "clustering": "clustering",
    "classification": "classification",
    "semantic-similarity": "sentence similarity",
    "pair-classification": "sentence similarity",
}

# Retrieval subtypes that have a more specific instruction, in order of precedence.
_RETRIEVAL_SUBTYPE_INSTRUCTION = {
    "Code retrieval": "code retrieval",
    "Claim verification": "fact checking",
    "Question answering": "question answering",
}


def _task_type_instruction(task_metadata: TaskMetadata) -> str:
    """Instruction from the simplified task type, refined by subtype and language for retrieval."""
    task_type = task_metadata.simplified_task_type
    if task_type == "retrieval":
        if any(lang in PROGRAMMING_LANGS for lang in task_metadata.languages):
            return "code retrieval"
        subtypes = task_metadata.task_subtypes or []
        for subtype, instruction in _RETRIEVAL_SUBTYPE_INSTRUCTION.items():
            if subtype in subtypes:
                return instruction
    return _TASK_TYPE_INSTRUCTION.get(task_type, "search result")


# (use_image, use_audio) -> modalities, n_parameters, memory_usage_mb of the loaded towers
_TOWERS = {
    (False, False): (["text"], 271_002_624, 1034),
    (True, False): (["text", "image", "video"], 438_760_448, 1674),
    (False, True): (["text", "audio"], 576_613_664, 2200),
    (True, True): (["text", "image", "audio", "video"], 744_371_992, 2840),
}


# Tokens one image expands to (256 soft tokens + begin/end markers) plus slack for the
# processor's wrapper tokens; text in front of an image is truncated to keep the whole
# image block inside max_seq_length (otherwise the processor raises a token-count mismatch).
_IMAGE_BLOCK_TOKENS = 260 + 24


class EmbeddingGemma2Wrapper(SentenceTransformerEncoderWrapper):
    """SentenceTransformer wrapper that applies the EmbeddingGemma 2 per-task recipe.

    Everything (prompt handling, multimodal inputs, MRL truncation via ``embed_dim``,
    normalisation) is inherited from ``SentenceTransformerEncoderWrapper``; this class only
    (1) loads just the requested modality towers, (2) sets ``max_seq_length`` for the
    loaded towers and (3) writes the per-task query / document templates into the batch text before
    delegating, because the document template ``title: {title} | text: {body}`` is not a
    plain prefix of mteb's ``"{title} {body}"`` corpus text.

    ``use_image`` / ``use_audio`` are the model's experiments: one ``ModelMeta`` covers all
    settings and the loaded towers decide the reported modalities and parameter count.

    Args:
        model_name: HF model id.
        revision: HF revision.
        use_image: Load the vision tower (image/video inputs). Used for MIEB.
        use_audio: Load the audio tower. Used for MAEB.
        media_first: Put the image/audio tokens *before* the prompt text instead of after
            it. The SentenceTransformer processor emits tokens in the order of the input
            dict's keys. ``False`` (text first) is the default; ``True`` reproduces the
            model-card MIEB(lite) run, whose internal pipeline inserted images at text
            position 0 (scores differ by ~0.4 Mean(TaskType), up to ±10 on single tasks).
        **kwargs: Forwarded to ``SentenceTransformerEncoderWrapper`` (e.g. ``embed_dim``).
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        revision: str | None = None,
        *,
        use_image: bool = False,
        use_audio: bool = False,
        media_first: bool = False,
        # video: 1 frame per second at 140 tokens per frame, the model card's defaults;
        # 56 frames fit its 8192-token context
        # https://huggingface.co/google/embeddinggemma-2
        fps: float | None = 1.0,
        max_frames: int | None = 56,
        **kwargs: Any,
    ) -> None:
        self.use_image, self.use_audio = use_image, use_audio
        self.media_first = media_first
        config_kwargs = {}  # drop the towers that are not needed
        if not use_image:
            config_kwargs["vision_config"] = None
        if not use_audio:
            config_kwargs["audio_config"] = None
        super().__init__(
            model_name,
            revision=revision,
            model_kwargs={"torch_dtype": "bfloat16"},
            config_kwargs=config_kwargs,
            target_sampling_rate=16000,
            fps=fps,
            max_frames=max_frames,
            **kwargs,
        )
        # Templates are written into the batch text in encode(). Disable both mteb's
        # model_prompts and the model's built-in "query"/"document" prompts, which
        # SentenceTransformer.encode_query/encode_document would otherwise prepend as well.
        self.model_prompts = {}
        self.model.prompts = {}
        self.model.default_prompt_name = None
        # One sequence length per tower configuration, the same for every task.
        if use_audio:
            self.model.max_seq_length = 1024
        elif use_image:
            self.model.max_seq_length = 380

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, meta: ModelMeta) -> None:
        """Report the modalities, parameter count and memory of the towers actually loaded."""
        modalities, n_parameters, memory_usage_mb = _TOWERS[
            self.use_image, self.use_audio
        ]
        self._mteb_model_meta = meta.model_copy(
            update={
                "modalities": modalities,
                "n_parameters": n_parameters,
                "memory_usage_mb": memory_usage_mb,
            },
            deep=True,
        )

    def _fit_before_image(self, text: str) -> str:
        """Truncate text that precedes an image so the image block still fits in max_seq_length."""
        budget = self.model.max_seq_length - _IMAGE_BLOCK_TOKENS
        ids = self.model.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return text
        return self.model.tokenizer.decode(ids[:budget])

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
        instruction = EMBEDDING_GEMMA_2_TASK_INSTRUCTIONS.get(
            name
        ) or _task_type_instruction(task_metadata)
        asymmetric = (
            task_metadata.type in {"Retrieval", "Any2AnyRetrieval"}
            or name in _MIEB_ASYMMETRIC_TASKS
        )
        is_document = asymmetric and prompt_type == PromptType.document
        prompt_on_images = name not in _MIEB_NO_PROMPT_ON_IMAGE_INPUTS

        # Attach audio/video collator (resamples raw audio to 16 kHz float32 mono with channel-averaging)
        _setup_modality_collator(
            inputs,
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
            target_sampling_rate=self.target_sampling_rate or 16000,
            max_samples=self.max_samples,
        )
        base_collate = inputs.collate_fn

        def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            batch = base_collate(rows)
            if "image" in batch and not prompt_on_images:
                return batch  # image inputs of this task get no prompt at all
            # Batch size from any column: text-less batches may be image, audio or video.
            n = len(next(iter(batch.values())))
            texts = batch.get("text") or [""] * n
            if is_document:
                bodies = batch.get("body", texts)  # corpus rows carry title/body
                titles = batch.get("title") or [""] * n
                batch["text"] = [
                    f"title: {t.strip() or 'none'} | text: {b}"
                    for t, b in zip(titles, bodies, strict=True)
                ]
            else:
                batch["text"] = [f"task: {instruction} | query: {t}" for t in texts]
            texts = batch.pop("text")
            if self.media_first:
                return {**batch, "text": texts}
            if "image" in batch:
                texts = [self._fit_before_image(t) for t in texts]
            return {"text": texts, **batch}

        original_class = inputs.__class__

        class _PreserveCollate(original_class):  # type: ignore[valid-type,misc]
            @property
            def collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
                return collate

            @collate_fn.setter
            def collate_fn(self, value: object) -> None:
                pass

        max_seq_length = self.model.max_seq_length
        if "video" in inputs.dataset.features:
            # one video is longer than the MIEB / MAEB lengths: use the full context
            self.model.max_seq_length = 8192
        try:
            inputs.__class__ = _PreserveCollate
            return super().encode(
                inputs,
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                prompt_type=prompt_type,
                **kwargs,
            )
        finally:
            inputs.__class__ = original_class
            inputs.collate_fn = base_collate
            self.model.max_seq_length = max_seq_length


embedding_gemma_2 = ModelMeta(
    loader=EmbeddingGemma2Wrapper,  # type: ignore[call-arg]
    name="google/embeddinggemma-2",
    model_type=["dense"],
    languages=MULTILINGUAL_EVALUATED_LANGUAGES,
    open_weights=True,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    release_date="2026-10-06",
    # Text-only towers (the default experiment); EmbeddingGemma2Wrapper updates
    # n_parameters / memory_usage_mb / modalities when use_image / use_audio are set.
    n_parameters=271_002_624,
    n_embedding_parameters=134_217_728,
    embed_dim=[768, 512, 256, 128],
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
    modalities=["text", "image", "audio", "video"],
    citation=EMBEDDING_GEMMA_CITATION,
    extra_requirements_groups=["embeddinggemma"],
)
