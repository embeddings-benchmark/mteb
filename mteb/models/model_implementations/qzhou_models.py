from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_implementations.bge_models import (
    bge_chinese_training_data,
    bge_full_data,
    bge_m3_training_data,
)
from mteb.models.model_implementations.e5_instruct import E5_MISTRAL_TRAINING_DATA
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

QZHOU_EMBEDDING_CITATION = """@misc{yu2025qzhouembeddingtechnicalreport,
      title={QZhou-Embedding Technical Report},
      author={Peng Yu and En Xu and Bin Chen and Haibiao Chen and Yinfei Xu},
      year={2025},
      eprint={2508.21632},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.21632},
}"""


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


qzhou_training_data = {
    "LCQMC",
    "PAWSX",
    "TNews",
    "Waimai",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MindSmallReranking",
    "STS12",
    "STS22.v2",
    "STSBenchmark",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "MLDR",
}

qzhou_zh_training_data = {"LCQMC", "PAWSX"}

QZhou_Embedding = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="Kingsoft-LLM/QZhou-Embedding",
    languages=["eng-Latn", "zho-Hans"],
    open_weights=True,
    revision="f1e6c03ee3882e7b9fa5cec91217715272e433b8",
    release_date="2025-08-24",
    n_parameters=7_070_619_136,
    memory_usage_mb=29070,
    embed_dim=3584,
    license="apache-2.0",
    max_tokens=8192,
    reference="https://huggingface.co/Kingsoft-LLM/QZhou-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/cfli/bge-full-data",
    training_datasets=bge_m3_training_data
    | bge_chinese_training_data
    | bge_full_data
    | E5_MISTRAL_TRAINING_DATA
    | qzhou_training_data,
    # Not in MTEB:
    # "FreedomIntelligence/Huatuo26M-Lite",
    # "infgrad/retrieval_data_llm",
    citation=QZHOU_EMBEDDING_CITATION,
)

QZhou_Embedding_Zh = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
    ),
    name="Kingsoft-LLM/QZhou-Embedding-Zh",
    languages=["zho-Hans"],
    open_weights=True,
    revision="0321ccb126413d1e49c5ce908e802b63d35f18e2",
    release_date="2025-09-28",
    n_parameters=7_575_747_328,
    memory_usage_mb=29431,
    embed_dim=1792,
    license="apache-2.0",
    max_tokens=8192,
    reference="http://huggingface.co/Kingsoft-LLM/QZhou-Embedding-Zh",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=qzhou_zh_training_data,
    citation=QZHOU_EMBEDDING_CITATION,
)
