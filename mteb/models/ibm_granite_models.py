from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

GRANITE_LANGUAGES = [
    "ara_Latn",
    "ces_Latn",
    "deu_Latn",
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "ita_Latn",
    "jpn_Latn",
    "kor_Latn",
    "nld_Latn",
    "por_Latn",
    "zho_Hant",
    "zho_Hans",
]

granite_training_data = {
    # Multilingual MC4
    # Multilingual Webhose
    # English Wikipedia
    # Multilingual Wikimedia
    "WikipediaRetrievalMultilingual": [],
    "WikipediaRerankingMultilingual": [],
    # Miracl Corpus (Title-Body)
    # Stack Exchange Duplicate questions (titles)
    # Stack Exchange Duplicate questions (titles)
    # Stack Exchange Duplicate questions (bodies)
    "StackOverflowDupQuestions": [],
    "AskUbuntuDupQuestions": [],
    # Stack Exchange (Title, Answer) pairs
    # Stack Exchange (Title, Body) pairs
    # Stack Exchange (Title, Body) pairs
    # Machine Translations of Stack Exchange Duplicate questions (titles)
    # Machine Translations of Stack Exchange (Title+Body, Answer) pairs
    "StackExchangeClusteringP2P": [],
    "StackExchangeClusteringP2P.v2": [],
    "StackExchangeClustering": [],
    "StackExchangeClustering.v2": [],
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
    "NQ": ["test"],
    "NQ-NL": ["test"],  # translation not trained on
    "NQHardNegatives": ["test"],
    # SQuAD2.0
    # HotpotQA
    "HotPotQA": ["test"],
    "HotPotQAHardNegatives": ["test"],
    "HotPotQA-PL": ["test"],  # translated from hotpotQA (not trained on)
    "HotpotQA-NL": ["test"],  # translated from hotpotQA (not trained on)
    # Fever
    "FEVER": ["test"],
    "FEVERHardNegatives": ["test"],
    "FEVER-NL": ["test"],  # translated from hotpotQA (not trained on)
    # PubMed
    # Multilingual Miracl Triples
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    # Multilingual MrTydi Triples
    "MrTidyRetrieval": ["train"],
    # Sadeeem Question Asnwering
    # DBPedia Title-Body Pairs
    "DBPedia": ["train"],
    "DBPedia-NL": ["train"],  # translated from hotpotQA (not trained on)
    # Synthetic: English Query-Wikipedia Passage
    # Synthetic: English Fact Verification
    # Synthetic: Multilingual Query-Wikipedia Passage
    # Synthetic: Multilingual News Summaries
    # IBM Internal Triples
    # IBM Internal Title-Body Pairs
}

granite_107m_multilingual = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-107m-multilingual",
        revision="47db56afe692f731540413c67dd818ff492277e7",
    ),
    name="ibm-granite/granite-embedding-107m-multilingual",
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="47db56afe692f731540413c67dd818ff492277e7",
    release_date="2024-12-18",
    n_parameters=107_000_000,
    memory_usage_mb=204,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
)

granite_278m_multilingual = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-278m-multilingual",
        revision="84e3546b88b0cb69f8078608a1df558020bcbf1f",
    ),
    name="ibm-granite/granite-embedding-278m-multilingual",
    languages=GRANITE_LANGUAGES,
    open_weights=True,
    revision="84e3546b88b0cb69f8078608a1df558020bcbf1f",
    release_date="2024-12-18",
    n_parameters=278_000_000,
    memory_usage_mb=530,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
)

granite_30m_english = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-30m-english",
        revision="eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5",
    ),
    name="ibm-granite/granite-embedding-30m-english",
    languages=["eng_Latn"],
    open_weights=True,
    revision="eddbb57470f896b5f8e2bfcb823d8f0e2d2024a5",
    release_date="2024-12-18",
    n_parameters=30_000_000,
    memory_usage_mb=58,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-30m-english",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
)

granite_125m_english = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="ibm-granite/granite-embedding-125m-english",
        revision="e48d3a5b47eaa18e3fe07d4676e187fd80f32730",
    ),
    name="ibm-granite/granite-embedding-125m-english",
    languages=["eng_Latn"],
    open_weights=True,
    revision="e48d3a5b47eaa18e3fe07d4676e187fd80f32730",
    release_date="2024-12-18",
    n_parameters=125_000_000,
    memory_usage_mb=238,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/ibm-granite/granite-embedding-125m-english",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets=granite_training_data,
)
