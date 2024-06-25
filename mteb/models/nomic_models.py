from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

nomic_embed = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        trust_remote_code=True,
        model_name="nomic-ai/nomic-embed-text-v1.5",
        revision=None,
    ),
    name="nomic-ai/nomic-embed-text-v1.5",
    languages=["eng-Latn"],
    open_source=True,
    revision=None,
    release_date="2024-02-10",  # first commit
)

if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(nomic_embed.name, nomic_embed.revision)
