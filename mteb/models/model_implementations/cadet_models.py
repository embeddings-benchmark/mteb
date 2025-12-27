from mteb.models.model_meta import ModelMeta
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader

from .bge_models import bge_m3_training_data

CADET_CITATION = """@article{tamber2025conventionalcontrastivelearningfalls,
    title={Conventional Contrastive Learning Often Falls Short: Improving Dense Retrieval with Cross-Encoder Listwise Distillation and Synthetic Data},
    author={Manveer Singh Tamber and Suleman Kazi and Vivek Sourabh and Jimmy Lin},
    journal={arXiv:2505.19274},
    year={2025}
}"""

cadet_training_data = {
    # we train with the corpora of FEVER, MSMARCO, and DBPEDIA. We only train with synthetic generated queries.
    # However, we do use queries from MSMARCO as examples for synthetic query generation.
    "MSMARCO",
    "DBPedia",
    # We distill from RankT5 which trains using MSMARCO + NQ.
    # source: https://arxiv.org/pdf/2210.10634
    "NQ",
    # We also distill from bge-rerankerv2.5-gemma2-lightweight, which utilizes the BGE-M3 dataset for training, along with Arguana, HotpotQA, and FEVER.
    # source: https://arxiv.org/pdf/2409.15700
    "FEVER",
    "HotpotQA",
    "ArguAna",
} | bge_m3_training_data


cadet_embed = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts={
            "query": "query: ",
            "document": "passage: ",
        },
    ),
    name="manveertamber/cadet-embed-base-v1",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="8056d118be37a566f20972a5f35cda815f6bc47e",
    open_weights=True,
    release_date="2025-05-11",
    n_parameters=109_000_000,
    memory_usage_mb=418,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/manveertamber/cadet-embed-base-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/manveertamber/cadet-dense-retrieval",
    # we provide the code to generate the training data
    public_training_data="https://github.com/manveertamber/cadet-dense-retrieval",
    training_datasets=cadet_training_data,
    adapted_from="intfloat/e5-base-unsupervised",
    citation=CADET_CITATION,
)
