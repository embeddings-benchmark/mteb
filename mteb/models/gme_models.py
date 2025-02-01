from __future__ import annotations

import logging

from mteb.model_meta import ModelMeta

logger = logging.getLogger(__name__)


gme_qwen2_vl_2b_instruct = ModelMeta(
    loader=None,
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="cfeb66885b598de483cc04eb08c7d9da534d7afe",
    release_date="2024-12-21",
    n_parameters=int(2.21 * 1e9),
    max_tokens=32768,
    embed_dim=1536,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        # Only annotating text data for now
        # source: https://arxiv.org/pdf/2412.16855
        "MSMARCO": ["train"],
        "MSMARCO.v2": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)

gme_qwen2_vl_7b_instruct = ModelMeta(
    loader=None,
    name="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    languages=["eng_Latn"],
    open_weights=True,
    revision="d42eca5a540526cfa982a349724b24b25c12a95e",
    release_date="2024-12-21",
    n_parameters=int(8.29 * 1e9),
    max_tokens=32768,
    embed_dim=3584,
    license="mit",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    reference="https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
    use_instructions=True,
    adapted_from=None,
    superseded_by=None,
    training_datasets={
        # Only annotating text data for now
        # source: https://arxiv.org/pdf/2412.16855
        "MSMARCO": ["train"],
        "MSMARCO.v2": ["train"],
    },
    public_training_code=None,
    public_training_data=None,
)
