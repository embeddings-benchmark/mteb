from __future__ import annotations

from mteb.models import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.types import PromptType

REASONIR_CITATION = """@article{shao2025reasonir,
      title={ReasonIR: Training Retrievers for Reasoning Tasks},
      author={Rulin Shao and Rui Qiao and Varsha Kishore and Niklas Muennighoff and Xi Victoria Lin and Daniela Rus and Bryan Kian Hsiang Low and Sewon Min and Wen-tau Yih and Pang Wei Koh and Luke Zettlemoyer},
      year={2025},
      journal={arXiv preprint arXiv:2504.20595},
      url={https://arxiv.org/abs/2504.20595},
}"""


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        # https://github.com/facebookresearch/ReasonIR/blob/0aac96269e455965949df16520fab72da68ffc22/evaluation/bright/configs/reasonir/economics.json#L3
        f"<|user|>\n{instruction}\n<|embed|>\n"
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else "<|embed|>\n"
    )


REASONIR_TRAINING_DATA = {
    # source, section D: https://arxiv.org/pdf/2504.20595
    "MSMARCO",
    "NQ",
    "FEVER",
    "HotpotQA",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
    "T2Reranking",
    "DuRetrieval",
    "QuoraRetrieval",
}
_prompts_dict = {
    "BrightBiologyRetrieval": {
        "query": "Given a Biology post, retrieve relevant passages that help answer the post"
    },
    "BrightEarthScienceRetrieval": {
        "query": "Given a Earth Science post, retrieve relevant passages that help answer the post"
    },
    "BrightEconomicsRetrieval": {
        "query": "Given a Economics post, retrieve relevant passages that help answer the post"
    },
    "BrightPsychologyRetrieval": {
        "query": "Given a Psychology post, retrieve relevant passages that help answer the post"
    },
    "BrightRoboticsRetrieval": {
        "query": "Given a Robotics post, retrieve relevant passages that help answer the post"
    },
    "BrightStackoverflowRetrieval": {
        "query": "Given a Stackoverflow post, retrieve relevant passages that help answer the post"
    },
    "BrightSustainableLivingRetrieval": {
        "query": "Given a Sustainable Living post, retrieve relevant passages that help answer the post"
    },
    "BrightPonyRetrieval": {
        "query": "Given a Pony question, retrieve relevant passages that help answer the question"
    },
    "BrightLeetcodeRetrieval": {
        "query": "Given a coding problem, retrieve relevant examples that help answer the problem",
    },
    "BrightAopsRetrieval": {
        "query": "Given a Math problem, retrieve relevant examples that help answer the problem"
    },
    "BrightTheoremQATheoremsRetrieval": {
        "query": "Given a Math problem, retrieve relevant theorems that help answer the problem",
    },
    "BrightTheoremQAQuestionsRetrieval": {
        "query": "Given a Math problem, retrieve relevant examples that help answer the problem",
    },
    "BrightBiologyLongRetrieval": {
        "query": "Given a Biology post, retrieve relevant documents that help answer the post"
    },
    "BrightEarthScienceLongRetrieval": {
        "query": "Given a Earth Science post, retrieve relevant documents that help answer the post"
    },
    "BrightEconomicsLongRetrieval": {
        "query": "Given a Economics post, retrieve relevant documents that help answer the post"
    },
    "BrightPsychologyLongRetrieval": {
        "query": "Given a Psychology post, retrieve relevant documents that help answer the post"
    },
    "BrightRoboticsLongRetrieval": {
        "query": "Given a Robotics post, retrieve relevant documents that help answer the post"
    },
    "BrightStackoverflowLongRetrieval": {
        "query": "Given a Stackoverflow post, retrieve relevant documents that help answer the post"
    },
    "BrightSustainableLivingLongRetrieval": {
        "query": "Given a Sustainable Living post, retrieve relevant documents that help answer the post"
    },
    "BrightPonyLongRetrieval": {
        "query": "Given a Pony question, retrieve relevant documents that help answer the question"
    },
}


ReasonIR_8B = ModelMeta(
    loader=InstructSentenceTransformerModel,
    loader_kwargs=dict(
        instruction_template=instruction_template,
        trust_remote_code=True,
        prompts_dict=_prompts_dict,
    ),
    name="ReasonIR/ReasonIR-8B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="c3d0690370ff4a8c3d3882d8dfa85c43650034fa",
    release_date="2025-04-29",
    n_parameters=7_500_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=131072,
    reference="https://huggingface.co/ReasonIR/ReasonIR-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch", "Transformers", "safetensors"],
    use_instructions=True,
    training_datasets=REASONIR_TRAINING_DATA,
    public_training_code="https://github.com/facebookresearch/ReasonIR/tree/main/training",
    public_training_data="https://huggingface.co/datasets/reasonir/reasonir-data",
    citation=REASONIR_CITATION,
)
