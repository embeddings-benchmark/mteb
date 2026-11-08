from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

all_minilm_l6_v2_genbais = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="lakinekaki/all-MiniLM-L6-v2-genbais",
    revision="0bc4516a09fe933a2d45d0a0103e46ca2c97e172",
    release_date="2026-03-01",
    languages=["eng-Latn"],
    n_parameters=22713216,
    n_embedding_parameters=11720448,
    memory_usage_mb=87,
    max_tokens=512,
    embed_dim=384,
    license="mit",
    open_weights=True,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    reference="https://huggingface.co/lakinekaki/all-MiniLM-L6-v2-genbais",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    training_datasets={
        "STSBenchmark",
        "AllNLI",
        "PAWS",
        "QuoraRetrieval",
        "NanoQuoraRetrieval",
        "TwentyNewsgroupsClustering",
    },
    public_training_code=None,
    public_training_data=None,
    model_type=["dense"],
    citation=None,
)
