from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

from .e5_models import ME5_TRAINING_DATA, model_prompts

E5_NL_CITATION = """
@misc{banar2025mtebnle5nlembeddingbenchmark,
  archiveprefix = {arXiv},
  author = {Nikolay Banar and Ehsan Lotfi and Jens Van Nooten and Cristina Arhiliuc and Marija Kliocaite and Walter Daelemans},
  eprint = {2509.12340},
  primaryclass = {cs.CL},
  title = {MTEB-NL and E5-NL: Embedding Benchmark and Models for Dutch},
  url = {https://arxiv.org/abs/2509.12340},
  year = {2025},
}
"""

e5_nl_small = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="clips/e5-small-trm-nl",
    model_type=["dense"],
    languages=["nld-Latn"],
    open_weights=True,
    revision="0243664a6c5e12eef854b091eb283e51833c3e9f",
    release_date="2025-09-23",
    n_parameters=40_800_000,
    memory_usage_mb=78,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/clips/e5-small-trm-nl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/ELotfi/e5-nl",
    public_training_data="https://huggingface.co/collections/clips/beir-nl",
    training_datasets=ME5_TRAINING_DATA,  # mMARCO-NL, HotpotQA-NL, FEVER-NL, and LLM generated data
    adapted_from="intfloat/multilingual-e5-small",
    citation=E5_NL_CITATION,
)

e5_nl_base = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="clips/e5-base-trm-nl",
    model_type=["dense"],
    languages=["nld-Latn"],
    open_weights=True,
    revision="6bd5722f236da48b4b8bcb28cc1fc478f7089956",
    release_date="2025-09-23",
    n_parameters=124_400_000,
    memory_usage_mb=237,
    embed_dim=768,
    license="mit",
    max_tokens=514,
    reference="https://huggingface.co/clips/e5-base-trm-nl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/ELotfi/e5-nl",
    public_training_data="https://huggingface.co/collections/clips/beir-nl",
    adapted_from="intfloat/multilingual-e5-base",
    training_datasets=ME5_TRAINING_DATA,  # mMARCO-NL, HotpotQA-NL, FEVER-NL, and LLM generated data
    citation=E5_NL_CITATION,
)

e5_nl_large = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="clips/e5-large-trm-nl",
    model_type=["dense"],
    languages=["nld-Latn"],
    open_weights=True,
    revision="683333f86ed9eb3699b5567f0fdabeb958d412b0",
    release_date="2025-09-23",
    n_parameters=355_000_000,
    memory_usage_mb=1355,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
    reference="https://huggingface.co/clips/e5-large-trm-nl",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/ELotfi/e5-nl",
    public_training_data="https://huggingface.co/collections/clips/beir-nl",
    training_datasets=ME5_TRAINING_DATA,  # mMARCO-NL, HotpotQA-NL, FEVER-NL, and LLM generated data
    adapted_from="intfloat/multilingual-e5-large",
    citation=E5_NL_CITATION,
)
