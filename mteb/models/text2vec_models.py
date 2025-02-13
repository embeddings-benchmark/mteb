"""Implementation of Text2Vec models"""

from __future__ import annotations

from mteb.model_meta import ModelMeta

# I couldn't find the large model on HF for some reason
text2vec_base_chinese = ModelMeta(
    name="shibing624/text2vec-base-chinese",
    languages=["zho-Hans"],
    open_weights=True,
    revision="183bb99aa7af74355fb58d16edf8c13ae7c5433e",
    release_date="2022-01-23",
    n_parameters=102 * 1e6,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/shibing624/text2vec-base-chinese",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Couldn't find it
    training_datasets={
        # source: https://huggingface.co/shibing624/text2vec-base-chinese
        # Not in MTEB
        # - shibing624/nli-zh-all/text2vec-base-chinese-sentence-dataset
        # (Could have overlaps I'm not aware of)
    },
    memory_usage_mb=390,
)

text2vec_base_chinese_paraphrase = ModelMeta(
    name="shibing624/text2vec-base-chinese-paraphrase",
    languages=["zho-Hans"],
    open_weights=True,
    revision="e90c150a9c7fb55a67712a766d6820c55fb83cdd",
    release_date="2023-06-19",
    n_parameters=118 * 1e6,
    memory_usage_mb=450,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from=None,
    public_training_code=None,
    public_training_data=None,  # Couldn't find it
    training_datasets={
        # source: https://huggingface.co/shibing624/text2vec-base-chinese
        # Not in MTEB
        # - shibing624/nli-zh-all/text2vec-base-chinese-paraphrase
        # (Could have overlaps I'm not aware of)
    },
)


text2vec_multi_langs = [
    "deu-Latn",  # German (de)
    "eng-Latn",  # English (en)
    "spa-Latn",  # Spanish (es)
    "fra-Latn",  # French (fr)
    "ita-Latn",  # Italian (it)
    "nld-Latn",  # Dutch (nl)
    "pol-Latn",  # Polish (pl)
    "por-Latn",  # Portuguese (pt)
    "rus-Cyrl",  # Russian (ru)
    "zho-Hans",  # Chinese (Simplified, zh)
]
text2vec_base_multilingual = ModelMeta(
    name="shibing624/text2vec-base-multilingual",
    languages=text2vec_multi_langs,
    open_weights=True,
    revision="6633dc49e554de7105458f8f2e96445c6598e9d1",
    release_date="2023-06-22",
    # While it can be loaded with SBERT, it has one suspicious file according to huggingface
    # So probably best not to.
    loader=None,
    n_parameters=118 * 1e6,
    memory_usage_mb=449,
    embed_dim=384,
    license="apache-2.0",
    max_tokens=256,
    reference="https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    superseded_by=None,
    adapted_from="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    public_training_code=None,
    public_training_data=None,  # Couldn't find it
    training_datasets={
        # source: https://huggingface.co/shibing624/text2vec-base-chinese
        # Not in MTEB
        # - shibing624/nli-zh-all/tree/main/text2vec-base-multilingual-dataset
        # # (Could have overlaps I'm not aware of)
    },
)
