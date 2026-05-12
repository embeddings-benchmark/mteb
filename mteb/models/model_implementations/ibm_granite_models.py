from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

GRANITE_EMBEDDING_CITATION = """@article{awasthy2025graniteembedding,
  title={Granite Embedding Models},
  author={Awasthy, Parul and Trivedi, Aashka and Li, Yulong and Bornea, Mihaela and Cox, David and Daniels, Abraham and Franz, Martin and Goodhart, Gabe and Iyer, Bhavani and Kumar, Vishwajeet and Lastras, Luis and McCarley, Scott and Murthy, Rudra and P, Vignesh and Rosenthal, Sara and Roukos, Salim and Sen, Jaydeep and Sharma, Sukriti and Sil, Avirup and Soule, Kate and Sultan, Arafat and Florian, Radu},
  journal={arXiv preprint arXiv:2502.20204},
  year={2025}
}"""

GRANITE_EMBEDDING_R2_CITATION = """@article{awasthy2025graniteembeddingr2models,
      title={Granite Embedding R2 Models},
      author={Parul Awasthy and Aashka Trivedi and Yulong Li and Meet Doshi and Riyaz Bhat and Vignesh P and Vishwajeet Kumar and Yushu Yang and Bhavani Iyer and Abraham Daniels and Rudra Murthy and Ken Barker and Martin Franz and Madison Lee and Todd Ward and Salim Roukos and David Cox and Luis Lastras and Jaydeep Sen and Radu Florian},
      year={2025},
      journal={arXiv preprint arXiv:2508.21085},
      url={https://arxiv.org/abs/2508.21085},
}"""

GRANITE_LANGUAGES = [
    "ara-Latn",
    "ces-Latn",
    "deu-Latn",
    "eng-Latn",
    "spa-Latn",
    "fra-Latn",
    "ita-Latn",
    "jpn-Latn",
    "kor-Latn",
    "nld-Latn",
    "por-Latn",
    "zho-Hant",
    "zho-Hans",
]

GRANITE_R2_LANGUAGES = [
    "ara-Arab",
    "aze-Latn",
    "ben-Beng",
    "bul-Cyrl",
    "cat-Latn",
    "ces-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "est-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hun-Latn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kor-Hang",
    "lav-Latn",
    "lit-Latn",
    "mar-Deva",
    "msa-Latn",
    "nld-Latn",
    "nor-Latn",
    "pol-Latn",
    "por-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "slk-Latn",
    "slv-Latn",
    "spa-Latn",
    "sqi-Latn",
    "srp-Cyrl",
    "swa-Latn",
    "swe-Latn",
    "tel-Telu",
    "tgl-Latn",
    "tha-Thai",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "uzb-Latn",
    "vie-Latn",
    "zho-Hans",
]

granite_training_data = {
    # Multilingual MC4
    # Multilingual Webhose
    # English Wikipedia
    # Multilingual Wikimedia
    "WikipediaRetrievalMultilingual",
    "WikipediaRerankingMultilingual",
    # Miracl Corpus (Title-Body)
    # Stack Exchange Duplicate questions (titles)
    # Stack Exchange Duplicate questions (titles)
    # Stack Exchange Duplicate questions (bodies)
    "StackOverflowDupQuestions",
    "AskUbuntuDupQuestions",
    # Stack Exchange (Title, Answer) pairs
    # Stack Exchange (Title, Body) pairs
    # Stack Exchange (Title, Body) pairs
    # Machine Translations of Stack Exchange Duplicate questions (titles)
    # Machine Translations of Stack Exchange (Title+Body, Answer) pairs
    "StackExchangeClusteringP2P",
    "StackExchangeClusteringP2P.v2",
    "StackExchangeClustering",
    "StackExchangeClustering.v2",
    # SearchQA
    # S2ORC (Title, Abstract)
    # WikiAnswers Duplicate question pairs
    # CCNews
    # XSum
    # SimpleWiki
    # Machine Translated Cross Lingual Parallel Corpora
    # SPECTER citation triplets
    # Machine Translations of SPECTER citation triplets
    # Natural Questions (NQ)
    "NQ",
    "NQ-NL",  # translation not trained on
    "NQHardNegatives",
    # SQuAD2.0
    # HotpotQA
    "HotPotQA",
    "HotPotQAHardNegatives",
    "HotPotQA-PL",  # translated from hotpotQA (not trained on)
    "HotpotQA-NL",  # translated from hotpotQA (not trained on)
    # Fever
    "FEVER",
    "FEVERHardNegatives",
    "FEVER-NL",  # translated from hotpotQA (not trained on)
    # PubMed
    # Multilingual Miracl Triples
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MIRACLReranking",
    # Multilingual MrTydi Triples
    "MrTidyRetrieval",
    # Sadeeem Question Answering
    # DBPedia Title-Body Pairs
    "DBPedia",
    "DBPedia-NL",  # translated from hotpotQA (not trained on)
    # Synthetic: English Query-Wikipedia Passage
    # Synthetic: English Fact Verification
    # Synthetic: Multilingual Query-Wikipedia Passage
    # Synthetic: Multilingual News Summaries
    # IBM Internal Triples
    # IBM Internal Title-Body Pairs
}

granite_multilingual_r2_training_data = {
    "DBPedia",
    "FEVER",
    "FEVERHardNegatives",
    "HotPotQA",
    "HotpotQAHardNegatives",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MrTidyRetrieval",
    "NQ",
    "NQHardNegatives",
    "SpartQA",
    "StackOverflowDupQuestions",
    "StatcanDialogueDatasetRetrieval",
    "TempReasonL1",
    "TempReasonL2Context",
    "TempReasonL2Fact",
    "TempReasonL2Pure",
    "TempReasonL3Context",
    "TempReasonL3Fact",
    "TempReasonL3Pure",
    "WinoGrande",
    # Also includes synthetic data and IBM Internal data
}
granite_107m_multilingual = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-107m-multilingual",
    model_type=["dense"],
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="47db56afe692f731540413c67dd818ff492277e7",
    release_date="2024-12-18",
    n_parameters=106994304,
    n_embedding_parameters=96_000_768,
    memory_usage_mb=204,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "ONNX",
        "safetensors",
    ],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_CITATION,
)

granite_278m_multilingual = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-278m-multilingual",
    model_type=["dense"],
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="84e3546b88b0cb69f8078608a1df558020bcbf1f",
    release_date="2024-12-18",
    n_parameters=278043648,
    n_embedding_parameters=192_001_536,
    memory_usage_mb=530,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "Transformers",
        "ONNX",
        "safetensors",
    ],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_CITATION,
)

granite_30m_english = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-30m-english",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5",
    release_date="2024-12-18",
    n_parameters=30295296,
    n_embedding_parameters=19_301_760,
    memory_usage_mb=58,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-30m-english",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_CITATION,
)

granite_125m_english = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-125m-english",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="e48d3a5b47eaa18e3fe07d4676e187fd80f32730",
    release_date="2024-12-18",
    n_parameters=124645632,
    n_embedding_parameters=38_603_520,
    memory_usage_mb=238,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-125m-english",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=[
        "Sentence Transformers",
        "PyTorch",
        "ONNX",
        "safetensors",
        "Transformers",
    ],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_CITATION,
)


granite_english_r2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-english-r2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="6e7b8ce0e76270394ac4669ba4bbd7133b60b7f9",
    release_date="2025-08-15",
    n_parameters=149014272,
    n_embedding_parameters=38682624,
    memory_usage_mb=284,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/ibm-granite/granite-embedding-english-r2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_R2_CITATION,
)

granite_small_english_r2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="ibm-granite/granite-embedding-small-english-r2",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="54a8d2616a0844355a5164432d3f6dafb37b17a3",
    release_date="2025-08-15",
    n_parameters=47662464,
    n_embedding_parameters=19341312,
    memory_usage_mb=91,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/ibm-granite/granite-embedding-small-english-r2",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
    citation=GRANITE_EMBEDDING_R2_CITATION,
)


granite_311m_multilingual_r2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="ibm-granite/granite-embedding-311m-multilingual-r2",
    revision="dba7b0ee9d789f330fecfb85df57699f9e7d9c42",
    release_date="2026-04-29",
    languages=GRANITE_R2_LANGUAGES,
    n_parameters=311664384,
    n_active_parameters_override=None,
    n_embedding_parameters=201332736,
    memory_usage_mb=594,
    max_tokens=8192,
    embed_dim=[128, 256, 384, 512, 768],
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=[
        "Sentence Transformers",
        "Transformers",
        "ONNX",
        "safetensors",
        "OpenVINO",
    ],
    reference="https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=granite_multilingual_r2_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=GRANITE_EMBEDDING_R2_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

granite_97m_multilingual_r2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs={},
    name="ibm-granite/granite-embedding-97m-multilingual-r2",
    revision="c61e626a6255c490879d0af885078b61929d51f6",
    release_date="2026-04-29",
    languages=GRANITE_R2_LANGUAGES,
    n_parameters=97441152,
    n_active_parameters_override=None,
    n_embedding_parameters=69120000,
    memory_usage_mb=186,
    max_tokens=8192,
    embed_dim=384,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=[
        "Sentence Transformers",
        "Transformers",
        "ONNX",
        "safetensors",
        "OpenVINO",
    ],
    reference="https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=granite_multilingual_r2_training_data,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=GRANITE_EMBEDDING_R2_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
