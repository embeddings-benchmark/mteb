from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .bge_models import bge_m3_training_data
from .e5_models import model_prompts

# All models below are trained on TyDi-QA (Arabic and English train), ARCD train and SQuAD train, plus
# machine-translated Arabic <-> English cross-lingual queries made from those same training questions.
# mteb has no task for TyDi-QA, ARCD or SQuAD themselves, so the tasks built from the same sources are listed.
# The models also inherit the training data of their base model through `adapted_from`.
_ARABIC_ENGLISH_RETRIEVAL_TRAINING_DATA = {
    "MrTidyRetrieval",  # Mr. TyDi is built on TyDi-QA; trained on TyDi-QA train (ar, en)
    "XQuADRetrieval",  # XQuAD is a translation of SQuAD v1.1 dev articles; trained on SQuAD train
    # Most queries of the MIRACL Arabic dev subset in MIRACLRetrievalHardNegatives.v2 are TyDi-QA Arabic
    # train questions, so the MIRACL tasks are listed explicitly (they are also inherited from the base models).
    "MIRACLRetrieval",
}

# The e5 v2 models are distilled from mzahran1/arabic-english-bge-m3-retrieval on the same training data,
# so the training data of the teacher (and of bge-m3, its base model) is included.
_DISTILLED_FROM_BGE_M3_TRAINING_DATA = (
    _ARABIC_ENGLISH_RETRIEVAL_TRAINING_DATA | bge_m3_training_data
)

arabic_english_e5_small_retrieval = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="mzahran1/arabic-english-e5-small-retrieval",
    model_type=["dense"],
    languages=["ara-Arab", "eng-Latn"],
    open_weights=True,
    revision="03b0f0eb8aefb21f27c51188c2d5740de6ce3073",
    release_date="2026-11-03",
    n_parameters=117_653_760,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=256,  # the checkpoint ships max_seq_length 256
    reference="https://huggingface.co/mzahran1/arabic-english-e5-small-retrieval",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval",
    public_training_data=None,
    adapted_from="intfloat/multilingual-e5-small",
    training_datasets=_ARABIC_ENGLISH_RETRIEVAL_TRAINING_DATA,
    citation=None,
)

# Fine-tuned from bge-m3, then weight-interpolated with it (WiSE-FT, 0.6 bge-m3 + 0.4 fine-tuned).
arabic_english_bge_m3_retrieval = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="mzahran1/arabic-english-bge-m3-retrieval",
    model_type=["dense"],
    languages=["ara-Arab", "eng-Latn"],
    open_weights=True,
    revision="467a29ba02658dbf772a7b87535cba06eb6525a3",
    release_date="2026-11-05",
    n_parameters=567_754_752,
    n_embedding_parameters=256_002_048,
    memory_usage_mb=2166,
    embed_dim=1024,
    license="mit",
    max_tokens=256,  # the checkpoint ships max_seq_length 256
    reference="https://huggingface.co/mzahran1/arabic-english-bge-m3-retrieval",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval",
    public_training_data=None,
    adapted_from="BAAI/bge-m3",
    training_datasets=_ARABIC_ENGLISH_RETRIEVAL_TRAINING_DATA,
    citation=None,
)

arabic_english_e5_base_retrieval = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="mzahran1/arabic-english-e5-base-retrieval",
    model_type=["dense"],
    languages=["ara-Arab", "eng-Latn"],
    open_weights=True,
    revision="90c4ceae29f504fc37ab2ca602e0cee17f751854",
    release_date="2026-11-05",
    n_parameters=278_043_648,
    n_embedding_parameters=192_001_536,
    memory_usage_mb=1061,
    embed_dim=768,
    license="mit",
    max_tokens=256,  # the checkpoint ships max_seq_length 256
    reference="https://huggingface.co/mzahran1/arabic-english-e5-base-retrieval",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval",
    public_training_data=None,
    adapted_from="intfloat/multilingual-e5-base",
    training_datasets=_DISTILLED_FROM_BGE_M3_TRAINING_DATA,
    citation=None,
)

arabic_english_e5_small_retrieval_v2 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="mzahran1/arabic-english-e5-small-retrieval-v2",
    model_type=["dense"],
    languages=["ara-Arab", "eng-Latn"],
    open_weights=True,
    revision="2427ed1677114b90a6f82d15abf77900d5a3c30c",
    release_date="2026-11-05",
    n_parameters=117_653_760,
    n_embedding_parameters=96_014_208,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=256,  # the checkpoint ships max_seq_length 256
    reference="https://huggingface.co/mzahran1/arabic-english-e5-small-retrieval-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval",
    public_training_data=None,
    adapted_from="intfloat/multilingual-e5-small",
    training_datasets=_DISTILLED_FROM_BGE_M3_TRAINING_DATA,
    citation=None,
)
