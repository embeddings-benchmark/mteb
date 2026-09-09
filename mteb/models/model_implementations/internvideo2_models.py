"""InternVideo2-CLIP model definition."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

INTERNVIDEO2_CITATION = """
@article{wang2024internvideo2,
  title={InternVideo2: Scaling Video Foundation Models for Multimodal Video Understanding},
  author={Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Xu, Jilan and Wang, Zun and others},
  journal={arXiv preprint arXiv:2403.15377},
  year={2024}
}"""

internvideo2_clip_1b_224p_f8 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    loader_kwargs=dict(num_frames=8, trust_remote_code=True),
    name="OpenGVLab/InternVideo2-CLIP-1B-224p-f8",  # discussions/3
    revision="2e08ec173489fc13a6e4ca2d9807927b73c653a3",
    release_date="2024-03-22",
    languages=["eng-Latn"],
    modalities=["video", "text"],
    model_type=["dense"],
    n_parameters=7_704_737_537,
    n_embedding_parameters=208_327_296,
    memory_usage_mb=29_391,
    max_tokens=80,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2",
    public_training_data=None,
    framework=["PyTorch"],
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation=INTERNVIDEO2_CITATION,
    extra_requirements_groups=["internvideo2"],
)
