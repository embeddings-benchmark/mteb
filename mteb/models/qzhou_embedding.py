
from mteb.model_meta import ModelMeta

my_model = ModelMeta(
    name="QZhou-Embedding",
    languages=["eng-Latn"， "zho-Hans"], 
    open_weights=True,
    revision="b43142d518d6e5251fd2d1e0a8741eef5c8b980a",
    release_date="2025-08-01",
    n_parameters=7_070_619_136,
    memory_usage_mb=14627,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/cfli/datasets",
    training_datasets={"bge-e5data": ["train"], "bge-full-data": ['train']},
)