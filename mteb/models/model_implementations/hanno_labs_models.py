from __future__ import annotations

from mteb.models.model_implementations.qwen3_models import q3e_instruct_loader
from mteb.models.model_meta import ModelMeta

# dinghy-law-0.6b-v1 is a legal-domain contrastive fine-tune of Qwen/Qwen3-Embedding-0.6B. It keeps the
# base architecture and interface exactly (last-token pooling, cosine similarity, query-side
# "Instruct: {instruction}\nQuery:" prefix with unprefixed documents), so it reuses the shared Qwen3
# instruct loader. Trained on public/permissively-licensed legal corpora; the GerDaLIR and BillSum splits
# below overlap MTEB tasks and are declared for zero-shot accounting.

DINGHY_LAW_CITATION = """@misc{dinghy-law-0.6b,
  title  = {dinghy-law-0.6b: a compact legal text-embedding model},
  author = {Solka, Stephen},
  year   = {2026},
  note   = {Hanno Labs / Clause Logic Inc.},
  howpublished = {\\url{https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1}}
}"""

# Sources used to train dinghy-law-0.6b-v1 that correspond to MTEB tasks (for train/test leakage accounting).
# GerDaLIR training data overlaps the GerDaLIR / GerDaLIRSmall corpora; BillSum (US) was used as a
# general-legal replay set. Other training sources (ECtHR, SCOTUS, LEDGAR, CAIL, GerLayQA, Indian statutes)
# do not correspond to MTEB tasks. Note: the Indian-statute training text shares public-domain statute
# documents with the AILAStatutes corpus, though none of AILA's queries or relevance labels were used.
# The Indian-statute training signal is publicly released as the CC-BY-4.0 dataset
# https://huggingface.co/datasets/Hanno-Labs/indian-ipc-statute-identification (Indian court judgments are public
# records under CC-BY-4.0; IPC statute text is a public-domain enactment; section citations masked as [STATUTE]).
# The training mix is permissively/publicly licensed -- no CC-BY-NC (non-commercial) sources are used.
training_data = {
    "GerDaLIR",
    "GerDaLIRSmall",
    "BillSumUS",
}

dinghy_law_0_6b = ModelMeta(
    loader=q3e_instruct_loader,
    # weights are stored bf16 (v1.1, revision fa421dfe: 1.19GB, byte-identical MTEB(Law,v1)=65.83 to the prior fp32
    # revision 8f63ca78 which stays in HF history); load in bf16 to match the base tier and the eval dtype.
    # Per-task leaderboard instructions (prompts_dict): isolated to be worth +3.65 (AILACasedocs) / +2.51
    # (AILAStatutes) nDCG over mteb's task-default instructions — the instruction is the dominant eval lever.
    loader_kwargs=dict(
        model_kwargs={"torch_dtype": "bfloat16"},
        prompts_dict={
            "AILACasedocs": "Retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
            "AILAStatutes": "Identify the most relevant statutes for the given situation.",
            "LegalSummarization": "Given a contract summary, retrieve the contract.",
            "LegalBenchConsumerContractsQA": "Retrieve the contract text most relevant to answering the given question.",
            "LegalBenchCorporateLobbying": "Given a bill title, retrieve the corresponding bill summary.",
            "GerDaLIRSmall": "Retrieve relevant legal case documents.",
            "LegalQuAD": "Retrieve relevant legal case documents.",
            "LeCaRDv2": "Retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        },
    ),
    name="Hanno-Labs/dinghy-law-0.6b-v1",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    open_weights=True,
    revision="fa421dfe45e21f3d676a707ffb04b05051bd9ae9",
    release_date="2026-07-13",
    n_parameters=596_826_112,
    n_embedding_parameters=155_309_056,
    memory_usage_mb=1138,  # bf16: 596,826,112 params x 2 bytes (matches base Qwen3-Emb-0.6B's 1136 + the 1024x1024 Dense)
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/Hanno-Labs/legal-retrieval-pairs-v1",
    training_datasets=training_data,
    citation=DINGHY_LAW_CITATION,
    adapted_from="Qwen/Qwen3-Embedding-0.6B",
)
