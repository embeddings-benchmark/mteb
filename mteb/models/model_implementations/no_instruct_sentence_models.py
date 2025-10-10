from collections.abc import Generator
from itertools import islice
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


# https://docs.python.org/3/library/itertools.html#itertools.batched
# Added in version 3.12.
def batched(iterable, n: int, *, strict: bool = False) -> Generator[tuple, None, None]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


class NoInstructModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        device = kwargs.pop("device", None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, **kwargs
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, **kwargs
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        sentences = [text for batch in inputs for text in batch["text"]]
        embeddings = []
        for batch in batched(sentences, batch_size):
            # Tokenize the batch
            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # The model is optimized to use the mean pooling for queries,
            # while the sentence / document embedding uses the [CLS] representation.
            if prompt_type == PromptType.query:
                # Mean pooling
                vectors = outputs.last_hidden_state * attention_mask.unsqueeze(2)
                pooled_vectors = vectors.sum(dim=1) / attention_mask.sum(
                    dim=-1, keepdim=True
                )
            else:
                # [CLS] token representation
                pooled_vectors = outputs.last_hidden_state[:, 0, :]

            # Append pooled vectors to result
            embeddings.append(pooled_vectors.cpu().detach().numpy())

        return np.concatenate(embeddings, axis=0)


no_instruct_small_v0 = ModelMeta(
    loader=NoInstructModel,
    name="avsolatorio/NoInstruct-small-Embedding-v0",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b38747000553d8268915c95a55fc87e707c9aadd",
    release_date="2024-05-01",  # first commit
    n_parameters=33_400_000,
    memory_usage_mb=127,
    max_tokens=512,
    embed_dim=384,
    license="mit",
    reference="https://huggingface.co/avsolatorio/NoInstruct-small-Embedding-v0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch"],
    use_instructions=False,
    adapted_from=None,
    superseded_by=None,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
