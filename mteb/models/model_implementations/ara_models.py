from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

arabic_triplet_matryoshka = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    model_type=["dense"],
    languages=["ara-Arab"],
    open_weights=True,
    revision="ed357f222f0b6ea6670d2c9b5a1cb93950d34200",
    release_date="2024-07-28",
    n_parameters=135193344,
    n_embedding_parameters=49_152_000,
    memory_usage_mb=516,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=768,
    reference="https://huggingface.co/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=False,
    public_training_code=None,
    adapted_from="aubmindlab/bert-base-arabertv02",
    public_training_data="akhooli/arabic-triplets-1m-curated-sims-len",
    training_datasets=set(
        #  "akhooli/arabic-triplets-1m-curated-sims-len"
    ),
    citation="""
    @inproceedings{nacar2025gate,
  author = {Nacar, Omer and Koubaa, Anis and Sibaee, Serry and Alhabashi, Yasser and Ammar, Adel and Boulila, Wadii},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  doi = {10.18653/2026.findings-emnlp.1952},
  pages = {41623–41634},
  publisher = {Association for Computational Linguistics},
  title = {GATE: General Arabic Text Embedding for Enhanced Semantic Textual Similarity with Matryoshka Representation Learning and Hybrid Loss Training},
  url = {https://doi.org/10.18653/2026.findings-emnlp.1952},
  year = {2026},
}""",
)

arabic_retrieval_v1 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    name="omarelshehy/Arabic-Retrieval-v1.0",
    model_type=["dense"],
    languages=["ara-Arab"],
    open_weights=True,
    revision="899f6e1b765915a72d5e4ace6bb2b221715550d8",
    release_date="2024-12-03",
    n_parameters=135_194_880,
    n_embedding_parameters=49_153_536,
    memory_usage_mb=516,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/omarelshehy/Arabic-Retrieval-v1.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch", "safetensors"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    adapted_from="aubmindlab/bert-base-arabertv02",
    training_datasets=None,  # the model card does not say what it was trained on
    citation=None,
)
