from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

NANOVDR_CITATION = """@article{nanovdr2026,
  title={NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval},
  author={Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal={arXiv preprint arXiv:2603.12824},
  year={2026}
}"""

nanovdr_s_multi = ModelMeta(
    loader=sentence_transformers_loader,
    name="nanovdr/NanoVDR-S-Multi",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "fra-Latn", "spa-Latn", "ita-Latn", "por-Latn"],
    open_weights=True,
    revision="b21574d7772ca26e22525543a2a6bf7081a95d8f",
    release_date="2026-02-26",
    modalities=["text"],
    n_parameters=69_000_000,
    n_embedding_parameters=1_572_864,
    memory_usage_mb=282,
    embed_dim=2048,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/nanovdr/NanoVDR-S-Multi",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/nanovdr/NanoVDR-Train",
    training_datasets={
        "VidoreTabfquadRetrieval",
        "VidoreDocVQARetrieval",
        "VidoreInfoVQARetrieval",
        "VidoreArxivQARetrieval",
    },
    citation=NANOVDR_CITATION,
)
