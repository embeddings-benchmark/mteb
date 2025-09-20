from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Helper function to perform last token pooling."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[
            torch.arange(batch_size, device=last_hidden.device), sequence_lengths
        ]
    return embedding


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    return (
        f"{instruction}\nQuery: "
        if (prompt_type is None or prompt_type == PromptType.query) and instruction
        else "Represent this passage\npassage: "
    )


class BMRetrieverWrapper(Wrapper):
    """Wrapper for the BMRetriever models."""

    def __init__(
        self,
        model_name_or_path: str,
        instruction_template: Callable[[str, PromptType | None], str],
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=kwargs.get("torch_dtype", torch.float32),
            trust_remote_code=kwargs.get("trust_remote_code", True),
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.instruction_template = instruction_template

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        all_embeddings = []

        instruction = self.get_task_instruction(task_name, prompt_type)
        instruction = self.format_instruction(instruction, prompt_type)
        prompted_sentences = [f"{instruction}{sentence}" for sentence in sentences]

        max_length = 512
        eos_token = self.tokenizer.eos_token or ""

        for i in tqdm(
            range(0, len(prompted_sentences), batch_size),
            desc=f"Encoding {'Queries' if prompt_type == PromptType.query else 'Passages'}...",
        ):
            batch_texts = prompted_sentences[i : i + batch_size]

            if eos_token:
                batch_texts = [text + eos_token for text in batch_texts]

                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=max_length - 1,  # for EOS token
                    return_token_type_ids=False,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True,
                )
                batch_dict["input_ids"] = [
                    ids + [self.tokenizer.eos_token_id]
                    for ids in batch_dict["input_ids"]
                ]

                batch_dict = self.tokenizer.pad(
                    batch_dict,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# Sources summarized from the paper's Table 8
BMRETRIEVER_TRAINING_DATA = {
    "PubMed": ["train"],  # https://huggingface.co/datasets/MedRAG/pubmed
    "raw_arxiv": ["train"],  # https://huggingface.co/datasets/mteb/raw_arxiv
    "medical_meadow_cord19": [
        "train"
    ],  # https://huggingface.co/datasets/medalpa/medical_meadow_cord19
    "textbooks": ["train"],  # https://huggingface.co/datasets/MedRAG/textbooks
    "statpearls": ["train"],  # https://huggingface.co/datasets/MedRAG/statpearls
    "LitCovid_BioCreative": [
        "train"
    ],  # https://huggingface.co/datasets/KushT/LitCovid_BioCreative
    "S2ORC": ["train"],  # https://github.com/allenai/s2orc
    "MSMARCO": [
        "train"
    ],  # https://huggingface.co/datasets/tevatro/msmarco-passage-corpus
    "BMRetriever/biomed_retrieval_dataset": [
        "train"
    ],  # https://huggingface.co/datasets/BMRetriever/biomed_retrieval_dataset
}

BMRetriever_410M = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name_or_path="BMRetriever/BMRetriever-410M",
        instruction_template=instruction_template,
        torch_dtype=torch.float32,
        trust_remote_code=True,
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
    similarity_fn_name="dot",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
)

BMRetriever_1B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name_or_path="BMRetriever/BMRetriever-1B",
        instruction_template=instruction_template,
        torch_dtype=torch.float32,
        trust_remote_code=True,
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
    similarity_fn_name="dot",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
)

BMRetriever_2B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name_or_path="BMRetriever/BMRetriever-2B",
        torch_dtype=torch.float32,
        instruction_template=instruction_template,
        trust_remote_code=True,
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
    similarity_fn_name="dot",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
)

BMRetriever_7B = ModelMeta(
    loader=partial(
        BMRetrieverWrapper,
        model_name_or_path="BMRetriever/BMRetriever-7B",
        instruction_template=instruction_template,
        torch_dtype=torch.float32,
    ),
    name="BMRetriever/BMRetriever-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="e3569bfbcfe3a1bc48c142e11a7b0f38e86065a3",
    release_date="2024-04-29",
    embed_dim=4096,
    n_parameters=7_110_660_096,
    memory_usage_mb=27124,
    max_tokens=32768,
    license="mit",
    reference="https://huggingface.co/BMRetriever/BMRetriever-7B",
    similarity_fn_name="dot",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=BMRETRIEVER_TRAINING_DATA,
)
