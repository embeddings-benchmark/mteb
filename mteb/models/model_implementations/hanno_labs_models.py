from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta

# dinghy-law-0.6b-v1: a legal-domain contrastive fine-tune of Qwen/Qwen3-Embedding-0.6B, keeping the base interface
# (last-token pooling, cosine, query-side "Instruct: {instruction}\nQuery:" prefix, unprefixed documents). Training
# data, licensing, and leakage accounting are on the model card.

DINGHY_LAW_CITATION = """@misc{dinghy-law-0.6b,
  title  = {dinghy-law-0.6b: a compact legal text-embedding model},
  author = {Solka, Stephen},
  year   = {2026},
  note   = {Hanno Labs / Clause Logic Inc.},
  howpublished = {\\url{https://huggingface.co/Hanno-Labs/dinghy-law-0.6b-v1}}
}"""

# training sources that correspond to MTEB tasks (for train/test leakage accounting)
training_data = {
    "GerDaLIR",
    "GerDaLIRSmall",
    "BillSumUS",
}

# Per-task query instruction (the dominant eval lever); wrapped at load time by the shared instruction_template below,
# and applied query-side only (apply_instruction_to_passages=False).
_instructions = {
    "AILACasedocs": "Retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
    "AILAStatutes": "Identify the most relevant statutes for the given situation.",
    "LegalSummarization": "Given a contract summary, retrieve the contract.",
    "LegalBenchConsumerContractsQA": "Retrieve the contract text most relevant to answering the given question.",
    "LegalBenchCorporateLobbying": "Given a bill title, retrieve the corresponding bill summary.",
    "GerDaLIRSmall": "Retrieve relevant legal case documents.",
    "LegalQuAD": "Retrieve relevant legal case documents.",
    "LeCaRDv2": "Retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
}

dinghy_law_0_6b = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template="Instruct: {instruction}\nQuery:",
        apply_instruction_to_passages=False,
        prompts_dict=_instructions,
        model_kwargs={"torch_dtype": "bfloat16"},
    ),
    name="Hanno-Labs/dinghy-law-0.6b-v1",
    model_type=["dense"],
    languages=["eng-Latn", "deu-Latn", "zho-Hans"],
    open_weights=True,
    revision="fa421dfe45e21f3d676a707ffb04b05051bd9ae9",
    release_date="2026-07-13",
    n_parameters=596_826_112,
    n_embedding_parameters=155_309_056,
    memory_usage_mb=1138,
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
