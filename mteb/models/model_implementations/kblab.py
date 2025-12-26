from mteb.models import sentence_transformers_loader
from mteb.models.model_meta import ModelMeta, ScoringFunction

sbert_swedish = ModelMeta(
    loader=sentence_transformers_loader,  # type: ignore[arg-type]
    name="KBLab/sentence-bert-swedish-cased",
    model_type=["dense"],
    languages=["swe-Latn"],
    open_weights=True,
    revision="6b5e83cd29c03729cfdc33d13b1423399b0efb5c",
    release_date="2023-01-11",
    n_parameters=124690944,
    memory_usage_mb=476,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=384,
    reference="https://huggingface.co/KBLab/sentence-bert-swedish-cased",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    adapted_from="sentence-transformers/all-mpnet-base-v2",
    citation="""@misc{rekathati2021introducing,  
  author = {Rekathati, Faton},  
  title = {The KBLab Blog: Introducing a Swedish Sentence Transformer},  
  url = {https://kb-labb.github.io/posts/2021-08-23-a-swedish-sentence-transformer/},  
  year = {2021}  
}""",
)
