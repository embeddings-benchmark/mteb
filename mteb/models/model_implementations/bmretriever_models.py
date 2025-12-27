from collections.abc import Callable
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from mteb.models import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.types import PromptType


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"{instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else "Represent this passage\npassage: "
    )


class BMRetrieverWrapper(InstructSentenceTransformerModel):
    def __init__(
        self,
        model_name: str,
        revision: str,
        instruction_template: str
        | Callable[[str, PromptType | None], str]
        | None = None,
        max_seq_length: int | None = None,
        apply_instruction_to_passages: bool = True,
        padding_side: str | None = None,
        add_eos_token: bool = False,
        prompts_dict: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.instruction_template = instruction_template
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.prompts_dict = prompts_dict

        tokenizer_params = {}
        if add_eos_token:
            tokenizer_params["add_eos_token"] = add_eos_token
        if max_seq_length is not None:
            tokenizer_params["model_max_length"] = max_seq_length
        if padding_side is not None:
            tokenizer_params["padding_side"] = padding_side
        kwargs.setdefault("tokenizer_args", {}).update(tokenizer_params)
        kwargs.setdefault("config_args", {}).update(revision=revision)

        transformer = Transformer(
            model_name,
            **kwargs,
        )
        pooling = Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode="lasttoken"
        )
        self.model = SentenceTransformer(modules=[transformer, pooling])


BMRETRIEVER_CITATION = """
@inproceedings{xu-etal-2024-bmretriever,
    title = "{BMR}etriever: Tuning Large Language Models as Better Biomedical Text Retrievers",
    author = "Xu, Ran and Shi, Wenqi and Yu, Yue and Zhuang, Yuchen and Zhu, Yanqiao and Wang, May Dongmei and Ho, Joyce C. and Zhang, Chao and Yang, Carl",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = "November",
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    pages = "22234--22254",
    url = "https://aclanthology.org/2024.emnlp-main.1241/"
}"""

# https://huggingface.co/datasets/BMRetriever/biomed_retrieval_dataset
BMRETRIEVER_TRAINING_DATA = {
    "FEVER",
    "MSMARCO",
    "NQ",
}

BMRetriever_410M = ModelMeta(
    loader=BMRetrieverWrapper,
    loader_kwargs=dict(
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-410M",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="e3569bfbcfe3a1bc48c142e11a7b0f38e86065a3",
    release_date="2024-04-29",
    embed_dim=1024,
    n_parameters=353_822_720,
    memory_usage_mb=1349,
    max_tokens=2048,
    license="mit",
    reference="https://huggingface.co/BMRetriever/BMRetriever-410M",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
    citation=BMRETRIEVER_CITATION,
)

BMRetriever_1B = ModelMeta(
    loader=BMRetrieverWrapper,
    loader_kwargs=dict(
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-1B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="1b758c5f4d3af48ef6035cc4088bdbcd7df43ca6",
    release_date="2024-04-29",
    embed_dim=2048,
    n_parameters=908_759_040,
    memory_usage_mb=3466,
    max_tokens=2048,
    license="mit",
    reference="https://huggingface.co/BMRetriever/BMRetriever-1B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
    citation=BMRETRIEVER_CITATION,
)

BMRetriever_2B = ModelMeta(
    loader=BMRetrieverWrapper,
    loader_kwargs=dict(
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-2B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="718179afd57926369c347f46eee616db81084941",
    release_date="2024-04-29",
    embed_dim=2048,
    n_parameters=2_506_172_416,
    memory_usage_mb=9560,
    max_tokens=8192,
    license="mit",
    reference="https://huggingface.co/BMRetriever/BMRetriever-2B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
    citation=BMRETRIEVER_CITATION,
)

BMRetriever_7B = ModelMeta(
    loader=BMRetrieverWrapper,
    loader_kwargs=dict(
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-7B",
    model_type=["dense"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="13e6adb9273c5f254e037987d6b44e9e4b005b9a",
    release_date="2024-04-29",
    embed_dim=4096,
    n_parameters=7_110_660_096,
    memory_usage_mb=27124,
    max_tokens=32768,
    license="mit",
    reference="https://huggingface.co/BMRetriever/BMRetriever-7B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
    citation=BMRETRIEVER_CITATION,
)
