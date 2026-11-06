from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from .e5_models import model_prompts

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
    max_tokens=512,
    reference="https://huggingface.co/mzahran1/arabic-english-e5-small-retrieval",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=True,
    public_training_code="https://github.com/mahmoudzahran01/arabic-english-retrieval-eval",
    public_training_data=None,
    adapted_from="intfloat/multilingual-e5-small",
    # Fine-tuned on TyDi-QA (Arabic and English train), ARCD train and SQuAD train, plus machine-translated
    # Arabic <-> English cross-lingual queries made from those same training questions.
    # mteb has no task for TyDi-QA, ARCD or SQuAD themselves, so the tasks built from the same sources are listed.
    # The model also inherits the multilingual-e5-small training data through `adapted_from`.
    training_datasets={
        "MrTidyRetrieval",  # Mr. TyDi is built on TyDi-QA; trained on TyDi-QA train (ar, en)
        "XQuADRetrieval",  # XQuAD is a translation of SQuAD v1.1 dev articles; trained on SQuAD train
        # Most queries of the MIRACL Arabic dev subset in MIRACLRetrievalHardNegatives.v2 are TyDi-QA Arabic
        # train questions, so the MIRACL tasks are listed explicitly (they are also inherited from multilingual-e5-small).
        "MIRACLRetrieval",
    },
    citation=None,
)
