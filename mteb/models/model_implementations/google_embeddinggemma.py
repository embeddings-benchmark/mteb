from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
# and MAEB were produced with one instruction per task (and, for text, sometimes a
# longer sequence length), so a task-type prompt dictionary does not reproduce them.
# Formatting:
#   * query / symmetric input ........ "task: {instruction} | query: {text}"
#   * document side of asymmetric tasks "title: {title} | text: {text}"
#                                        (title = "none" when the corpus has no title)
#   * asymmetric = retrieval tasks, based on metadata and on an explicit list for
#     MIEB below.
#   * MIEB: max_seq_length 380 for every task (image tokens included); for the tasks in
#     _MIEB_NO_PROMPT_ON_IMAGE_INPUTS any input containing an image gets no prompt.
#   * MAEB: max_seq_length 1024 for every task (audio tokens included). Raw audio
#     inputs are resampled to 16 kHz float32 mono with channel-averaging prior to
#     tokenization. All audio documents and retrieval corpora are embedded with prefix
#     "title: none | text: <|audio|>".
#   * bf16, mean pooling, L2-normalised, 768d (MRL truncation via ``embed_dim``).
# Tasks outside these benchmarks fall back to a simplified task-type instruction
# and 512 tokens (380 with vision, 1024 with audio).

EMBEDDING_GEMMA_2_TASK_RECIPE: dict[str, tuple[str, int]] = {
    "AfriSentiClassification": ("classification", 512),
    "AILAStatutes": ("sentence similarity", 512),
    "AlloProfClusteringS2S.v2": ("code retrieval", 2048),
    "AlloprofReranking": ("search result", 2048),
    "AmazonCounterfactualClassification": ("classification", 512),
    "AppsRetrieval": ("code retrieval", 2048),
    "ArguAna": ("fact checking", 512),
    "ArmenianParaphrasePC": ("classification", 1024),
    "AROCocoOrder": ("question answering", 380),
    "AROFlickrOrder": ("clustering", 380),
    "AROVisualAttribution": ("search result", 380),
    "AROVisualRelation": ("classification", 380),
    "ArXivHierarchicalClusteringP2P": ("clustering", 512),
    "ArXivHierarchicalClusteringS2S": ("clustering", 512),
    "AskUbuntuDupQuestions": ("search result", 2048),
    "Banking77Classification": ("classification", 512),
    "BeijingOpera": ("classification", 1024),
    "BelebeleRetrieval": ("question answering", 2048),
    "BibleNLPBitextMining": ("search result", 2048),
    "BigPatentClustering.v2": ("sentence similarity", 2048),
    "BiorxivClusteringP2P.v2": ("code retrieval", 2048),
    "BIOSSES": ("search result", 2048),
    "BirdCLEF": ("sentence similarity", 1024),
    "BLINKIT2IMultiChoice": ("sentence similarity", 380),
    "BornholmBitextMining": ("sentence similarity", 512),
    "BrazilianToxicTweetsClassification": ("classification", 1024),
    "BUCC.v2": ("sentence similarity", 512),
    "BulgarianStoreReviewSentimentClassfication": ("classification", 512),
    "CataloniaTweetClassification": ("code retrieval", 2048),
    "CEDRClassification": ("classification", 512),
    "CIFAR100ZeroShot": ("sentence similarity", 380),
    "CIRRIT2IRetrieval": ("clustering", 380),
    "ClimateFEVERHardNegatives": ("question answering", 1024),
    "ClothoT2ARetrieval": ("search result", 1024),
    "CLSClusteringP2P.v2": ("code retrieval", 2048),
    "CodeEditSearchRetrieval": ("code retrieval", 512),
    "CodeFeedbackMT": ("code retrieval", 2048),
    "CodeFeedbackST": ("code retrieval", 2048),
    "CodeSearchNetCCRetrieval": ("code retrieval", 1024),
    "CodeSearchNetRetrieval": ("code retrieval", 2048),
    "CodeTransOceanContest": ("code retrieval", 2048),
    "CodeTransOceanDL": ("code retrieval", 1024),
    "COIRCodeSearchNetRetrieval": ("code retrieval", 2048),
    "CommonLanguageAgeDetection": ("clustering", 1024),
    "CommonVoiceMini21T2ARetrieval": ("sentence similarity", 1024),
    "Core17InstructionRetrieval": ("question answering", 2048),
    "CosQA": ("code retrieval", 1024),
    "Country211": ("clustering", 380),
    "Country211ZeroShot": ("code retrieval", 380),
    "CovidRetrieval": ("search result", 1024),
    "CQADupstackGamingRetrieval": ("search result", 2048),
    "CQADupstackUnixRetrieval": ("question answering", 512),
    "CREMA_D": ("clustering", 1024),
    "CREMA_DClustering": ("fact checking", 1024),
    "CREMADPairClassification": ("clustering", 1024),
    "CSFDSKMovieReviewSentimentClassification": ("classification", 2048),
    "CTKFactsNLI": ("classification", 512),
    "CUB200I2IRetrieval": ("clustering", 380),
    "CVBenchCount": ("code retrieval", 380),
    "CVBenchDepth": ("sentence similarity", 380),
    "CVBenchDistance": ("classification", 380),
    "CVBenchRelation": ("classification", 380),
    "CyrillicTurkicLangClassification": ("clustering", 1024),
    "CzechProductReviewSentimentClassification": ("classification", 1024),
    "DalajClassification": ("classification", 512),
    "DBpediaClassification": ("classification", 1024),
    "DiaBlaBitextMining": ("sentence similarity", 512),
    "DTD": ("question answering", 380),
    "EstonianValenceClassification": ("classification", 512),
    "EuroSAT": ("sentence similarity", 380),
    "FaroeseSTS": ("sentence similarity", 512),
    "Fashion200kI2TRetrieval": ("search result", 380),
    "FER2013ZeroShot": ("question answering", 380),
    "FEVERHardNegatives": ("search result", 1024),
    "FGVCAircraftZeroShot": ("question answering", 380),
    "FilipinoShopeeReviewsClassification": ("classification", 512),
    "FinancialPhrasebankClassification": ("sentence similarity", 512),
    "FinParaSTS": ("code retrieval", 2048),
    "FiQA2018": ("question answering", 2048),
    "FleursT2ARetrieval": ("search result", 1024),
    "FloresBitextMining": ("search result", 1024),
    "Food101ZeroShot": ("question answering", 380),
    "FSD2019Kaggle": ("classification", 1024),
    "GermanSTSBenchmark": ("sentence similarity", 512),
    "GigaSpeechT2ARetrieval": ("question answering", 1024),
    "GreekLegalCodeClassification": ("question answering", 2048),
    "GTSRB": ("question answering", 380),
    "GTZANAudioReranking": ("code retrieval", 1024),
    "GTZANGenre": ("fact checking", 1024),
    "GujaratiNewsClassification": ("clustering", 512),
    "HagridRetrieval": ("sentence similarity", 512),
    "HALClusteringS2S.v2": ("clustering", 512),
    "HatefulMemesI2TRetrieval": ("question answering", 380),
    "HotpotQAHardNegatives": ("question answering", 1024),
    "IEMOCAPGender": ("clustering", 1024),
    "ImageCoDe": ("fact checking", 380),
    "ImageNetDog15Clustering": ("clustering", 380),
    "ImdbClassification": ("classification", 2048),
    "IN22GenBitextMining": ("sentence similarity", 512),
    "IndicCrosslingualSTS": ("search result", 1024),
    "IndicGenBenchFloresBitextMining": ("sentence similarity", 1024),
    "IndicLangClassification": ("clustering", 1024),
    "IndonesianIdClickbaitClassification": ("classification", 512),
    "indonli": ("sentence similarity", 1024),
    "InfoSeekIT2TRetrieval": ("fact checking", 380),
    "IsiZuluNewsClassification": ("fact checking", 512),
    "ItaCaseholdClassification": ("clustering", 512),
    "JamAltArtistA2ARetrieval": ("fact checking", 1024),
    "JamAltLyricA2TRetrieval": ("search result", 1024),
    "JSICK": ("sentence similarity", 1024),
    "KorHateSpeechMLClassification": ("search result", 512),
    "KorSarcasmClassification": ("question answering", 512),
    "KurdishSentimentClassification": ("classification", 1024),
    "LegalBenchCorporateLobbying": ("search result", 2048),
    "LEMBPasskeyRetrieval": ("sentence similarity", 2048),
    "MacedonianTweetSentimentClassification": ("classification", 512),
    "MACST2ARetrieval": ("sentence similarity", 1024),
    "MalteseNewsClassification": ("clustering", 2048),
    "MasakhaNEWSClassification": ("clustering", 1024),
    "MasakhaNEWSClusteringS2S": ("clustering", 512),
    "MassiveIntentClassification": ("classification", 1024),
    "MassiveScenarioClassification": ("classification", 2048),
    "MedrxivClusteringP2P.v2": ("clustering", 512),
    "MedrxivClusteringS2S.v2": ("clustering", 2048),
    "MInDS14": ("question answering", 1024),
    "MindSmallReranking": ("clustering", 512),
    "MIRACLRetrievalHardNegatives": ("fact checking", 2048),
    "MLQARetrieval": ("search result", 2048),
    "MridinghamTonic": ("question answering", 1024),
    "MTOPDomainClassification": ("classification", 2048),
    "MultiEURLEXMultilabelClassification": ("question answering", 1024),
    "MultiHateClassification": ("sentence similarity", 512),
    "NepaliNewsClassification": ("clustering", 512),
    "News21InstructionRetrieval": ("classification", 512),
    "NIGHTSI2IRetrieval": ("clustering", 380),
    "NMSQAPairClassification": ("sentence similarity", 1024),
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
    "OVENIT2TRetrieval": ("question answering", 380),
    "OxfordPets": ("fact checking", 380),
    "OxfordPetsZeroShot": ("fact checking", 380),
    "PAC": ("classification", 512),
    "PatchCamelyon": ("clustering", 380),
    "PawsXPairClassification": ("question answering", 1024),
    "PlscClusteringP2P.v2": ("clustering", 512),
    "PoemSentimentClassification": ("sentence similarity", 512),
    "PolEmo2.0-OUT": ("classification", 512),
    "PpcPC": ("sentence similarity", 1024),
    "PunjabiNewsClassification": ("sentence similarity", 2048),
    "RavdessZeroshot": ("clustering", 1024),
    "RESISC45": ("sentence similarity", 380),
    "Robust04InstructionRetrieval": ("question answering", 1024),
    "RomaniBibleClustering": ("fact checking", 2048),
    "RP2kI2IRetrieval": ("clustering", 380),
    "RTE3": ("sentence similarity", 512),
    "RuBQReranking": ("fact checking", 512),
    "ScalaClassification": ("sentence similarity", 2048),
    "SCIDOCS": ("question answering", 1024),
    "SemRel24STS": ("sentence similarity", 1024),
    "SentimentAnalysisHindi": ("classification", 512),
    "SIB200ClusteringS2S": ("sentence similarity", 512),
    "SIBFLEURS": ("clustering", 1024),
    "SICK-R": ("sentence similarity", 512),
    "SinhalaNewsClassification": ("clustering", 512),
    "SiswatiNewsClassification": ("search result", 1024),
    "SlovakMovieReviewSentimentClassification": ("classification", 512),
    "SpartQA": ("question answering", 512),
    "SpeechCommandsZeroshotv0.02": ("sentence similarity", 1024),
    "SpokenSQuADT2ARetrieval": ("fact checking", 1024),
    "SprintDuplicateQuestions": ("search result", 512),
    "StackExchangeClustering.v2": ("clustering", 512),
    "StackExchangeClusteringP2P.v2": ("clustering", 512),
    "StackOverflowQA": ("code retrieval", 2048),
    "StanfordCarsZeroShot": ("question answering", 380),
    "StatcanDialogueDatasetRetrieval": ("search result", 2048),
    "STS12": ("sentence similarity", 512),
    "STS13": ("sentence similarity", 512),
    "STS13VisualSTS": ("sentence similarity", 380),
    "STS14": ("sentence similarity", 512),
    "STS15": ("sentence similarity", 1024),
    "STS15VisualSTS": ("sentence similarity", 380),
    "STS17": ("sentence similarity", 512),
    "STS17MultilingualVisualSTS": ("sentence similarity", 380),
    "STS22.v2": ("sentence similarity", 1024),
    "STSB": ("sentence similarity", 512),
    "STSBenchmark": ("sentence similarity", 1024),
    "STSBenchmarkMultilingualVisualSTS": ("sentence similarity", 380),
    "STSES": ("sentence similarity", 512),
    "SummEvalSummarization.v2": ("sentence similarity", 2048),
    "SUN397": ("sentence similarity", 380),
    "SwahiliNewsClassification": ("clustering", 1024),
    "SwednClusteringP2P": ("clustering", 1024),
    "SwissJudgementClassification": ("question answering", 2048),
    "SyntheticText2SQL": ("code retrieval", 2048),
    "T2Reranking": ("fact checking", 512),
    "Tatoeba": ("sentence similarity", 1024),
    "TempReasonL1": ("search result", 512),
    "TERRa": ("sentence similarity", 512),
    "TinyImageNetClustering": ("fact checking", 380),
    "Touche2020Retrieval.v3": ("question answering", 1024),
    "ToxicConversationsClassification": ("classification", 512),
    "TRECCOVID": ("search result", 2048),
    "TswanaNewsClassification": ("search result", 2048),
    "TweetSentimentExtractionClassification": ("classification", 2048),
    "TweetTopicSingleClassification": ("clustering", 512),
    "TwentyNewsgroupsClustering.v2": ("clustering", 2048),
    "TwitterHjerneRetrieval": ("search result", 2048),
    "TwitterSemEval2015": ("sentence similarity", 512),
    "TwitterURLCorpus": ("sentence similarity", 512),
    "UrbanSound8KT2ARetrieval": ("fact checking", 1024),
    "VehicleSoundClustering": ("sentence similarity", 1024),
    "VidoreDocVQARetrieval": ("question answering", 380),
    "VidoreInfoVQARetrieval": ("search result", 380),
    "VidoreShiftProjectRetrieval": ("search result", 380),
    "VidoreSyntheticDocQAAIRetrieval": ("search result", 380),
    "VidoreTabfquadRetrieval": ("question answering", 380),
    "VidoreTatdqaRetrieval": ("search result", 380),
    "VisualNewsI2TRetrieval": ("fact checking", 380),
    "VisualSTS-b-Multilingual": ("sentence similarity", 380),
    "VisualSTS17Multilingual": ("sentence similarity", 380),
    "VoxCelebSA": ("classification", 1024),
    "VoxPopuliAccentPairClassification": ("clustering", 1024),
    "VoxPopuliGenderClustering": ("sentence similarity", 1024),
    "VoxPopuliLanguageID": ("clustering", 1024),
    "VoyageMMarcoReranking": ("fact checking", 2048),
    "VQA2IT2TRetrieval": ("clustering", 380),
    "WebLINXCandidatesReranking": ("classification", 2048),
    "WebQAT2ITRetrieval": ("search result", 380),
    "WikiCitiesClustering": ("clustering", 1024),
    "WikiClusteringP2P.v2": ("classification", 1024),
    "WikipediaRerankingMultilingual": ("sentence similarity", 1024),
    "WikipediaRetrievalMultilingual": ("search result", 512),
    "WinoGrande": ("search result", 512),
    "Winoground": ("search result", 380),
    "WITT2IRetrieval": ("question answering", 380),
    "XM3600T2IRetrieval": ("question answering", 380),
    "XNLI": ("sentence similarity", 512),
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
    (1) loads just the requested modality towers, (2) sets the per-task ``max_seq_length``
    and (3) writes the per-task query / document templates into the batch text before
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
            **kwargs,
        )
        # Templates are written into the batch text in encode(). Disable both mteb's
        # model_prompts and the model's built-in "query"/"document" prompts, which
        # SentenceTransformer.encode_query/encode_document would otherwise prepend as well.
        self.model_prompts = {}
        self.model.prompts = {}
        self.model.default_prompt_name = None

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

    def _fit_before_image(self, text: str, seq_len: int) -> str:
        """Truncate text that precedes an image so the image block still fits in seq_len."""
        budget = seq_len - _IMAGE_BLOCK_TOKENS
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
        instruction, seq_len = EMBEDDING_GEMMA_2_TASK_RECIPE.get(name) or (
            _TASK_TYPE_INSTRUCTION.get(
                task_metadata.simplified_task_type, "search result"
            ),
            1024 if self.use_audio else (380 if self.use_image else 512),
        )
        self.model.max_seq_length = seq_len
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
                texts = [self._fit_before_image(t, seq_len) for t in texts]
            return {"text": texts, **batch}

        original_class = inputs.__class__

        class _PreserveCollate(original_class):  # type: ignore[valid-type,misc]
            @property
            def collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
                return collate

            @collate_fn.setter
            def collate_fn(self, value: object) -> None:
                pass

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
