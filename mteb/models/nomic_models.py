from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

emb = model.encode(["Hello, world!"])

# !pip install einops


from mteb.model_meta import ModelMeta


def sentence_transformers_loader(
    model_name: str, revision: str, **kwargs
) -> SentenceTransformer:
    return SentenceTransformer(
        model_name_or_path=model_name,
        revision=revision,
        trust_remote_code=True,  # required
        **kwargs,
    )


nomic_embed = ModelMeta(
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_source=True,
    revision=None,
    release_date="2024-02-10",  # first commit
)
