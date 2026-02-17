"""
MTEB ModelMeta for ManiacLabs/miniac-embed.

LEAF-distilled embedding model: E5-small-unsupervised backbone, mxbai-embed-large teacher.
"""

from mteb.models import ModelMeta, sentence_transformers_loader

# LEAF: Knowledge Distillation of Text Embedding Models
LEAF_CITATION = """@misc{mdbr_leaf,
  title={LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations},
  author={Robin Vujanic and Thomas Rueckstiess},
  year={2025},
  eprint={2509.12539},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2509.12539}
}"""

# E5: Weakly-Supervised Contrastive Pre-training (student backbone)
E5_CITATION = """@article{wang2022e5,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022},
  url={https://arxiv.org/abs/2212.03533}
}"""

MINIAC_EMBED_CITATION = (
    "This model uses LEAF distillation (Vujanic & Rueckstiess, 2025) with an "
    "E5-small-unsupervised backbone (Wang et al., 2022).\n\n"
    "LEAF: " + LEAF_CITATION + "\n\n"
    "E5: " + E5_CITATION
)

model_prompts = {"query": "Represent this sentence for searching relevant passages: "}

MINIAC_EMBED_TRAINING_DATASETS = {
    "fineweb",
    "cc_news",
    "english-words-definitions",
    "amazon-qa",
    "msmarco",
    "PubMedQA",
    "trivia_qa",
    "LoTTE",
}

miniac_embed = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="ManiacLabs/miniac-embed",
    model_type=["dense"],
    revision="0fe5413163ce75cf13e6351b39a8b6f321b64e79",
    release_date="2026-02-13",
    languages=["eng-Latn"],
    open_weights=True,
    framework=[
        "PyTorch",
        "Sentence Transformers",
    ],
    n_parameters=33_360_000,
    n_embedding_parameters=11_917_056,
    memory_usage_mb=127,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/ManiacLabs/miniac-embed",
    similarity_fn_name="cosine",
    use_instructions=True,
    adapted_from="intfloat/e5-small-unsupervised",
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=MINIAC_EMBED_TRAINING_DATASETS,
    citation=MINIAC_EMBED_CITATION,
)
