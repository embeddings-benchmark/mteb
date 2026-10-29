"""InternVideo2-CLIP model definition."""

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

INTERNVIDEO2_CITATION = """
@inbook{wang2024internvideo2,
  author = {Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Wang, Zun and Shi, Yansong and Jiang, Tianxiang and Li, Songze and Xu, Jilan and Zhang, Hongjie and Huang, Yifei and Qiao, Yu and Wang, Yali and Wang, Limin},
  booktitle = {Computer Vision – ECCV 2024},
  doi = {10.1007/978-3-031-73013-9_23},
  isbn = {9783031730139},
  issn = {1611-3349},
  month = {Nov},
  pages = {396–416},
  publisher = {Springer Nature Switzerland},
  title = {InternVideo2: Scaling Foundation Models for Multimodal Video Understanding},
  url = {https://doi.org/10.1007/978-3-031-73013-9_23},
  year = {2024},
}"""

internvideo2_clip_1b_224p_f8 = ModelMeta(
    loader=SentenceTransformerEncoderWrapper,
    # 8 frames: encoded in the checkpoint name
    # https://github.com/OpenGVLab/InternVideo
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
