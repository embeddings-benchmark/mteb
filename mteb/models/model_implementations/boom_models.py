"""
Models from ICT-TIME-and-Querit organization.

这个文件应该位置在：mteb/models/model_implementations/ict_time.py
"""

from functools import partial

from mteb.model_meta import ModelMeta
from mteb.models.loader import sentence_transformers_loader


# ===== BOOM_4B_v1 模型注册 =====
ict_time_boom_4b_v1 = ModelMeta(
    name="ICT-TIME-and-Querit/BOOM_4B_v1",
    revision="33fb345468120e37c81eed2369aefe08b8f8222b",  
    open_weights=True,
    languages=["deu", "ita", "ara", "fas", "fra", "hin", "spa", "zho", "ara", "ben", "eng", "fin", "ind", "jpn", "kor", "rus", "swh", "tel", "tha"], 
    release_date="2026-01-31",  # 根据实际情况调整
    n_parameters=4021774336,  # 4 billion parameters
    memory_usage_mb=7671,  # 根据实际情况调整
    embed_dim=2560,  # 根据模型实际嵌入维度调整
    max_tokens=32768,  # 根据模型实际支持的最大令牌数调整
    license="apache-2.0", 
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    reference="https://huggingface.co/ICT-TIME-and-Querit/BOOM_4B_v1",
)
