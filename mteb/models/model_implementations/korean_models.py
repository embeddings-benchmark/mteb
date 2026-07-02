"""Korean(-focused) embedding models, registered for MTEB(kor, v2).

These community models had Korean retrieval/STS/clustering results but no ModelMeta
in the registry, so they were invisible on the leaderboard. Metadata (revision,
n_parameters, embed_dim, max_tokens, license, HF repo creation date, base model) was
fetched from the Hugging Face Hub. Loaders mirror each model's base family
(bge-m3 / multilingual-e5 / arctic-embed / plain sentence-transformers).
"""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

# E5-style query/passage prefixes (multilingual-e5 family)
E5_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "passage: ",
}
# arctic-embed-v2 style: instruction prefix on the query only
ARCTIC_QUERY_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "",
}

KOR_EN = ["kor-Hang", "eng-Latn"]

# --------------------------------------------------------------------------- #
# BAAI/bge-m3 Korean fine-tunes (dense, no query instruction, cosine)
# --------------------------------------------------------------------------- #
dragonkue_bge_m3_ko = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="dragonkue/BGE-m3-ko",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="7074d66aa46562342193ca4feb3d89bf9dad71b4",
    release_date="2024-09-17",
    n_parameters=567_754_752,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8194,
    memory_usage_mb=2166,
    n_embedding_parameters=256_002_048,
    reference="https://huggingface.co/dragonkue/BGE-m3-ko",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="BAAI/bge-m3",
    public_training_code=None,
    public_training_data=None,
    # Confirmed by the model author: the Korean fine-tuning data contains no
    # mteb datasets; empty set so the bge-m3 base data is still inherited via
    # adapted_from.
    training_datasets=set(),
)

kure_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="nlpai-lab/KURE-v1",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="d14c8a9423946e268a0c9952fecf3a7aabd73bd9",
    release_date="2024-12-18",
    n_parameters=567_754_752,
    embed_dim=1024,
    license="mit",
    max_tokens=8194,
    memory_usage_mb=2166,
    n_embedding_parameters=256_002_048,
    reference="https://huggingface.co/nlpai-lab/KURE-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="BAAI/bge-m3",
    public_training_code=None,
    public_training_data=None,
    # Trained on ~2M Korean query-document-hard-negative pairs; the model card and
    # the KURE GitHub repo do not name the source datasets, so unknown.
    training_datasets=None,
)

upskyy_bge_m3_korean = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="upskyy/bge-m3-korean",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="069ae0627320935e4b2879522edbb54650b59bf5",
    release_date="2024-08-09",
    n_parameters=567_754_752,
    embed_dim=1024,
    license=None,
    max_tokens=8194,
    memory_usage_mb=2166,
    n_embedding_parameters=256_002_048,
    reference="https://huggingface.co/upskyy/bge-m3-korean",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="BAAI/bge-m3",
    public_training_code=None,
    public_training_data=None,
    # Model card: "korsts and kornli finetuning model from BAAI/bge-m3".
    # KorNLI has no mteb task; base bge-m3 data is inherited via adapted_from.
    training_datasets={"KorSTS"},
)

# --------------------------------------------------------------------------- #
# intfloat/multilingual-e5 Korean fine-tunes (query:/passage: prefixes)
# --------------------------------------------------------------------------- #
dragonkue_e5_small_ko = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(model_prompts=E5_PROMPTS),
    name="dragonkue/multilingual-e5-small-ko",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="bba4cb8e3299d445c04f62bf62e2b26461d03715",
    release_date="2025-05-11",
    n_parameters=117_653_760,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    memory_usage_mb=449,
    n_embedding_parameters=96_014_208,
    reference="https://huggingface.co/dragonkue/multilingual-e5-small-ko",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="intfloat/multilingual-e5-small",
    public_training_code=None,
    public_training_data=None,
    # Fine-tuned on the same data as dragonkue/snowflake-arctic-embed-l-v2.0-ko
    # (AI Hub Korean MRC corpora — none are mteb datasets); empty set so the
    # multilingual-e5 base data is still inherited via adapted_from.
    training_datasets=set(),
)

dragonkue_koen_e5_tiny = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(model_prompts=E5_PROMPTS),
    name="exp-models/dragonkue-KoEn-E5-Tiny",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="292c09c78c71a3f00ed56ee0d1ed9f0d39182fc9",
    release_date="2025-05-13",
    n_parameters=37_517_184,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=512,
    memory_usage_mb=143,
    n_embedding_parameters=15_877_632,
    reference="https://huggingface.co/exp-models/dragonkue-KoEn-E5-Tiny",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="intfloat/multilingual-e5-small",
    public_training_code=None,
    public_training_data=None,
    # Model card: fine-tuned on the same data as
    # dragonkue/snowflake-arctic-embed-l-v2.0-ko (AI Hub Korean MRC corpora —
    # none are mteb datasets).
    training_datasets=set(),
)

koe5 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(model_prompts=E5_PROMPTS),
    name="nlpai-lab/KoE5",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="bc6d284c60fe5a973e74c1751b92594c9f581213",
    release_date="2024-09-24",
    n_parameters=559_890_432,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    memory_usage_mb=2136,
    n_embedding_parameters=256_002_048,
    reference="https://huggingface.co/nlpai-lab/KoE5",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="intfloat/multilingual-e5-large",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/nlpai-lab/ko-triplet-v1.0",
    # Trained on nlpai-lab/ko-triplet-v1.0 (~700k Korean triplets, open data);
    # its dataset card does not document which source datasets it aggregates,
    # so possible mteb-task overlap is unknown.
    training_datasets=None,
)

# --------------------------------------------------------------------------- #
# Snowflake arctic-embed-l-v2.0 Korean fine-tune
# --------------------------------------------------------------------------- #
dragonkue_arctic_l_v2_0_ko = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(model_prompts=ARCTIC_QUERY_PROMPTS),
    name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    model_type=["dense"],
    languages=KOR_EN,
    open_weights=True,
    revision="55ec6e9358a56d56af759bc8372e970caf8c305f",
    release_date="2025-03-07",
    n_parameters=567_754_752,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=8192,
    memory_usage_mb=2166,
    n_embedding_parameters=256_002_048,
    reference="https://huggingface.co/dragonkue/snowflake-arctic-embed-l-v2.0-ko",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    adapted_from="Snowflake/snowflake-arctic-embed-l-v2.0",
    public_training_code=None,
    public_training_data=None,
    # Model card: fine-tuned on AI Hub Korean MRC corpora (admin-document /
    # news / book / numeric / finance-law MRC) — none are mteb datasets; empty
    # set so the arctic-embed base data is still inherited via adapted_from.
    training_datasets=set(),
)

# --------------------------------------------------------------------------- #
# jhgan/ko-sroberta-multitask (KLUE RoBERTa-base SBERT, mean pooling, cosine)
# --------------------------------------------------------------------------- #
ko_sroberta_multitask = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="jhgan/ko-sroberta-multitask",
    model_type=["dense"],
    languages=["kor-Hang"],
    open_weights=True,
    revision="8fca7c9c98c26599be0e14b9916b11a756a26f19",
    release_date="2022-03-02",
    n_parameters=110_618_626,
    embed_dim=768,
    license=None,
    max_tokens=512,
    memory_usage_mb=422,
    n_embedding_parameters=24_576_000,
    reference="https://huggingface.co/jhgan/ko-sroberta-multitask",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    adapted_from="klue/roberta-base",
    public_training_code=None,
    public_training_data=None,
    # Model card: multi-task trained on the KorSTS and KorNLI training sets
    # (KorNLI has no mteb task).
    training_datasets={"KorSTS"},
)
