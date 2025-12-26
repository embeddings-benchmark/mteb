from mteb.models.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

PAWAN_EMBD_CITATION = """@misc{medhi2025pawanembd,
    title={PawanEmbd-68M: Distilled Embedding Model},
    author={Medhi, D.},
    year={2025},
    url={https://huggingface.co/dmedhi/PawanEmbd-68M}
}"""

pawan_embd_68m = ModelMeta(
    loader=sentence_transformers_loader,
    name="dmedhi/PawanEmbd-68M",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="32f295145802bdbd65699ad65fd27d2a5b69a909",
    release_date="2025-12-08",
    n_parameters=68_000_000,
    memory_usage_mb=260,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/dmedhi/PawanEmbd-68M",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    adapted_from="ibm-granite/granite-embedding-278m-multilingual",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    use_instructions=False,
    training_datasets={
        "AllNLI",
    },
    citation=PAWAN_EMBD_CITATION,
)
