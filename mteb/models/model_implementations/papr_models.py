from __future__ import annotations

from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_implementations.qwen3_models import multilingual_langs
from mteb.models.model_meta import ModelMeta

# Per-task instructions (task_prompts.json in each model repo, identical for both sizes).
papr_embed_code_prompts = {
    "AppsRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "COIRCodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeFeedbackMT": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeFeedbackST": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeSearchNetCCRetrieval": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeTransOceanContest": "Given a piece for code, retrieval semantically similar code",
    "CodeTransOceanDL": "Given a piece for code, retrieval semantically similar code",
    "CosQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "SyntheticText2SQL": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "CodeEditSearchRetrieval": "Given a piece of code, retrieval code that in the ",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
}

papr_embed_code_instruction_template = "Instruct: {instruction}\nQuery:"

# Both models are fine-tuned from Qwen3-Embedding, so they inherit its training sets ...
_base_training = {
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "NQ",
    "MSMARCO",
    "HotpotQA",
    "FEVER",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "CodeSearchNet",
}
# ... plus the train splits used here. Only train splits; no test split was used.
_coir_train_4b = {
    "AppsRetrieval",
    "CodeFeedbackMT",
    "CodeSearchNetCCRetrieval",
    "COIRCodeSearchNetRetrieval",
    "StackOverflowQA",
    "SyntheticText2SQL",
    "CosQA",
    "CodeTransOceanDL",
}  # CodeFeedbackST and CodeTransOceanContest were not trained on
# Both sizes also trained on lightonai/CodeEditSearchTrain (commit message -> diff, derived from
# bigcode/commitpackft and decontaminated against cassanof/CodeEditSearch), and on the CoIR CodeSearchNet
# train split, which shares its source with CodeSearchNetRetrieval. Both evals are declared as seen.
_code_near = {"CodeEditSearchRetrieval", "CodeSearchNetRetrieval"}
papr_embed_code_4b_training_datasets = (
    _base_training | _coir_train_4b | _code_near | {"SciFact"}
)
papr_embed_code_0b6_training_datasets = papr_embed_code_4b_training_datasets | {
    "CodeFeedbackST",
    "CodeTransOceanContest",
}

_common = dict(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=papr_embed_code_instruction_template,
        apply_instruction_to_passages=False,  # queries carry the instruction, documents are bare
        prompts_dict=papr_embed_code_prompts,
        # Both checkpoints support 32k positions (0.6B: 32768, 4B: 40960); long-document tasks such as
        # SWEbenchCodeRetrieval need it (21% of its documents exceed 8k tokens).
        max_seq_length=32768,
    ),
    model_type=["dense"],
    languages=multilingual_langs,
    open_weights=True,
    release_date="2026-10-22",
    max_tokens=32768,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "safetensors", "Transformers"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
)

papr_embed_code_1_4b = ModelMeta(
    name="papr-ai/papr-embed-code-1-4b",
    revision="2abaf9b8b8dbcda87259cbd274c75c40f085d8c7",
    n_parameters=4021774336,
    n_embedding_parameters=388262400,
    memory_usage_mb=15342,  # fp32 reference precision
    embed_dim=2560,
    reference="https://huggingface.co/papr-ai/papr-embed-code-1-4b",
    adapted_from="Qwen/Qwen3-Embedding-4B",
    training_datasets=papr_embed_code_4b_training_datasets,
    **_common,
)

papr_embed_code_1_0b6 = ModelMeta(
    name="papr-ai/papr-embed-code-1-0.6b",
    revision="43f82f3adbde6477d16d88746f018294fac3514c",
    n_parameters=595776512,
    n_embedding_parameters=155309056,
    memory_usage_mb=2273,  # fp32 reference precision
    embed_dim=1024,
    reference="https://huggingface.co/papr-ai/papr-embed-code-1-0.6b",
    adapted_from="Qwen/Qwen3-Embedding-0.6B",
    training_datasets=papr_embed_code_0b6_training_datasets,
    **_common,
)
