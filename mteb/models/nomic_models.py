from __future__ import annotations

from functools import partial

from sentence_transformers import SentenceTransformer

from mteb.model_meta import ModelMeta


class SentenceTransformerWithNormalization(SentenceTransformer):
    def encode(self, sentences, *args, **kwargs):
        if "normalize_embeddings" not in kwargs:
            kwargs["normalize_embeddings"] = True

        return super().encode(sentences, *args, **kwargs)


def sentence_transformers_loader(
    model_name: str, revision: str | None, **kwargs
) -> SentenceTransformer:
    return SentenceTransformerWithNormalization(
        model_name_or_path=model_name, revision=revision, **kwargs
    )


nomic_embed = ModelMeta(
    loader=partial(  # type: ignore
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
    emb = mdl.encode(["test"], convert_to_tensor=True)
