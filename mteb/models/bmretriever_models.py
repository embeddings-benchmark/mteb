from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"{instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else "Represent this passage\npassage: "
    )


class BMRetrieverWrapper(InstructSentenceTransformerWrapper):
    def __init__(
        self,
        model_name: str,
        instruction_template: Callable[[str, PromptType | None], str] | None = None,
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

        transformer = Transformer(
            model_name,
            **kwargs,
        )
        pooling = Pooling(
            transformer.get_word_embedding_dimension(), pooling_mode="lasttoken"
        )
        self.model = SentenceTransformer(modules=[transformer, pooling])


# https://huggingface.co/datasets/BMRetriever/biomed_retrieval_dataset
BMRETRIEVER_TRAINING_DATA = {
    "FEVER": ["train"],
    "MSMARCO": ["train"],
    "NQ": ["train"],
}

BMRetriever_410M = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name="BMRetriever/BMRetriever-410M",
        config_args={"revision": "e3569bfbcfe3a1bc48c142e11a7b0f38e86065a3"},
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-410M",
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
)

BMRetriever_1B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name="BMRetriever/BMRetriever-1B",
        config_args={"revision": "1b758c5f4d3af48ef6035cc4088bdbcd7df43ca6"},
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-1B",
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
)

BMRetriever_2B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name="BMRetriever/BMRetriever-2B",
        config_args={"revision": "718179afd57926369c347f46eee616db81084941"},
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-2B",
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
)

BMRetriever_7B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name="BMRetriever/BMRetriever-7B",
        config_args={"revision": "13e6adb9273c5f254e037987d6b44e9e4b005b9a"},
        model_args={"torch_dtype": torch.float32},
        instruction_template=instruction_template,
        padding_side="left",
        add_eos_token=True,
        apply_instruction_to_passages=True,
    ),
    name="BMRetriever/BMRetriever-7B",
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
)
