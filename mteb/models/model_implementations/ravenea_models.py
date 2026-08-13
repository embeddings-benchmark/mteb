from mteb.models.model_implementations.clip_models import CLIPModel
from mteb.models.model_meta import ModelMeta, ScoringFunction

RAVENEA_CITATION = r"""
@inproceedings{li2026ravenea,
  author = {Jiaang Li and Yifei Yuan and Wenyan Li and Mohammad Aliannejadi and Daniel Hershcovich and Anders S{\o}gaard and Ivan Vuli{\'c} and Wenxuan Zhang and Paul Pu Liang and Yang Deng and Serge Belongie},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  title = {{RAVENEA}: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding},
  url = {https://openreview.net/forum?id=4zAbkxQ23i},
  year = {2026},
}
"""


ravenea_clip_vit_large_patch14 = ModelMeta(
    loader=CLIPModel,
    name="jaagli/ravenea-clip-vit-large-patch14",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="890d85d3539a21fab2bb349d4874d5dfef5dd3ec",
    release_date="2026-02-13",
    modalities=["image", "text"],
    n_parameters=427_616_513,
    n_embedding_parameters=37_945_344,
    memory_usage_mb=1631,
    max_tokens=77,
    embed_dim=768,
    license=None,
    open_weights=True,
    public_training_code="https://github.com/yfyuan01/RAVENEA",
    public_training_data="https://huggingface.co/datasets/jaagli/ravenea",
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/jaagli/ravenea-clip-vit-large-patch14",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets={"RAVENEAI2TRetrieval"},
    adapted_from="openai/clip-vit-large-patch14",
    citation=RAVENEA_CITATION,
)
