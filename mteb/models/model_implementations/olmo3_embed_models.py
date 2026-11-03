from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.types import OutputDType

# Training sources that are (the train split of) an MTEB task; data card:
# https://github.com/petegordon/olmo3-embed/blob/main/docs/data-card.md
# The rest of the training data (StackExchange with the 12 CQADupStack forums
# excluded, S2ORC title-citation pairs, PubMedQA) is not an MTEB task. Hard
# negatives were mined and filtered with Qwen/Qwen3-Embedding-0.6B as a scorer
# (no distillation of its embeddings).
OLMO3_EMBED_TRAINING_DATA = {
    "MSMARCO",
    "HotpotQA",
    "NQ",
    "FEVER",
    "FiQA2018",
    "NFCorpus",
    "SciFact",
    "QuoraRetrieval",
}

olmo3_embed_7b = ModelMeta(
    # LoRA fine-tune of OLMo-3-7B (merged): causal attention, last-token pooling
    # on an EOS the tokenizer appends, queries take "Instruct: ...\nQuery: ".
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template="Instruct: {instruction}\nQuery: ",
        apply_instruction_to_passages=False,
        max_seq_length=512,
        model_kwargs={"torch_dtype": "bfloat16"},
    ),
    name="petegordon/olmo3-embed-7b",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="ad1008e0c5857f40e472077b7ff43450f89b66d9",
    release_date="2026-10-14",
    n_parameters=6_887_272_448,
    n_embedding_parameters=410_738_688,
    memory_usage_mb=13137,
    embed_dim=4096,
    max_tokens=512,
    license="apache-2.0",
    reference="https://huggingface.co/petegordon/olmo3-embed-7b",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code="https://github.com/petegordon/olmo3-embed",
    public_training_data="https://github.com/petegordon/olmo3-embed/blob/main/docs/data-card.md",
    training_datasets=OLMO3_EMBED_TRAINING_DATA,
    adapted_from="allenai/Olmo-3-1025-7B",
    contacts=["petegordon"],
    output_dtypes=[OutputDType.BF16],
)
