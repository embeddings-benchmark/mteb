from mteb.models import sentence_transformers_loader
from mteb.models.model_meta import ModelMeta, ScoringFunction

xlmr_scandi = ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[arg-type]
    name="KFST/XLMRoberta-en-da-sv-nb",
    model_type=["dense"],
    languages=["swe-Latn", "nob-Latn", "nno-Latn", "dan-Latn", "eng-Latn"],
    open_weights=True,
    revision="d40c10ca7b1e68b5a8372f2d112dac9eb3279df1",
    release_date="2022-02-22",
    n_parameters=278043648,
    memory_usage_mb=1061,
    embed_dim=768,
    license="not specified",
    max_tokens=512,
    reference="https://huggingface.co/KFST/XLMRoberta-en-da-sv-nb",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="FacebookAI/xlm-roberta-base",
)
