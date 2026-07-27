from __future__ import annotations

from mteb.models.model_implementations.qwen3_models import q3e_instruct_loader
from mteb.models.model_meta import ModelMeta

# Fetched verbatim from https://arxiv.org/bibtex/2508.07995 (v5); the copy in
# the model card predates the revision and lists an older author set.
DIVER_CITATION = """@misc{sun2026divermultistageapproachreasoningintensive,
      title={DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval},
      author={Duolin Sun and Meixiu Long and Dan Yang and Junjie Wang and Yecheng Luo and Yue Shen and Jian Wang and Hualei Zhou and Chunxiao Guo and Peng Wei and Jiahai Wang and Jinjie Gu},
      year={2026},
      eprint={2508.07995},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.07995},
}"""

DIVER_RETRIEVER_4B = ModelMeta(
    # A Qwen3-Embedding-4B fine-tune, so it keeps that interface: last-token
    # pooling, cosine similarity, and an "Instruct: ...\nQuery:" prefix on
    # queries only (verified against its config_sentence_transformers.json).
    loader=q3e_instruct_loader,
    name="AQ-MedAI/Diver-Retriever-4B-1020",
    model_type=["dense"],
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="3984377d297f4c3bcf5d625ea7b48feea7f58c57",
    release_date="2025-10-20",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=7670,
    embed_dim=2560,
    max_tokens=40960,
    license="apache-2.0",
    reference="https://huggingface.co/AQ-MedAI/Diver-Retriever-4B-1020",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/AQ-MedAI/Diver",
    public_training_data=None,
    # The model card lists reasonir/reasonir-data among its training sets, and
    # the hard-query half of that dataset reconstructs its positive documents
    # from xlangai/BRIGHT — so the BRIGHT corpora are part of this model's
    # training data too. The other listed sets (MATH_qCoT..., truehealth/medqa,
    # AQ-MedAI/PRGB-ZH) are not MTEB tasks.
    training_datasets={
        "BrightAopsRetrieval",
        "BrightBiologyRetrieval",
        "BrightEarthScienceRetrieval",
        "BrightEconomicsRetrieval",
        "BrightLeetcodeRetrieval",
        "BrightPonyRetrieval",
        "BrightPsychologyRetrieval",
        "BrightRoboticsRetrieval",
        "BrightStackoverflowRetrieval",
        "BrightSustainableLivingRetrieval",
        "BrightTheoremQAQuestionsRetrieval",
        "BrightTheoremQATheoremsRetrieval",
    },
    adapted_from="Qwen/Qwen3-Embedding-4B",
    citation=DIVER_CITATION,
)
