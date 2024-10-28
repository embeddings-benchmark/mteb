from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

jina_clip_v1 = ModelMeta(
    loader=partial(
        sentence_transformers_loader,
        model_name="jinaai/jina-clip-v1",
    ),
    name="jinaai/jina-clip-v1",
    languages=["eng_Latn"],
    open_source=True,
    revision="1cbe5e8b11ea3728df0b610d5453dfe739804aa9",
    release_date="2024-05-30",
)


if __name__ == "__main__":
    import mteb

    mdl = mteb.get_model(jina_clip_v1.name, jina_clip_v1.revision)
    emb = mdl.get_text_embeddings(["Hello, world!"])
