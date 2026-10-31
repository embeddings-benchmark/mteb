from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

INSTRUCTION = "Given a query, retrieve relevant passages that answer the query"


class EtsuEncoder(AbsEncoder):
    """Etsu embedding encoder (Qwen3.5 text backbone).

    Query text is prefixed with the instruction template used during training;
    passages are encoded raw. Embeddings use last-token pooling followed by L2
    normalisation, matching the released encoding protocol.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_length: int = 2048,
        batch_size: int = 16,
        **kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True,
                dtype=torch.bfloat16
                if self.device.startswith("cuda")
                else torch.float32,
            )
            .to(self.device)
            .eval()
        )
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def _format(texts: list[str], is_query: bool) -> list[str]:
        if not is_query:
            return texts
        return [f"Instruct: {INSTRUCTION}\nQuery: {t}" for t in texts]

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode texts into L2-normalized embeddings."""
        import torch
        import torch.nn.functional as F

        sentences: list[str] = []
        for batch in inputs:
            if "query" in batch:
                key = "query"
            elif "text" in batch:
                key = "text"
            elif "body" in batch:
                key = "body"
            else:
                raise KeyError(f"unsupported batch keys: {sorted(batch)}")
            for item in batch[key]:
                sentences.extend(item if isinstance(item, list) else [item])

        is_query = getattr(prompt_type, "value", prompt_type) == "query"
        texts = self._format(sentences, is_query)

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**encoded, use_cache=False, return_dict=True)
                attention_mask = encoded["attention_mask"]
                left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
                if left_padding:
                    embeddings = output.last_hidden_state[:, -1]
                else:
                    lengths = attention_mask.sum(dim=1) - 1
                    rows = torch.arange(
                        output.last_hidden_state.shape[0],
                        device=output.last_hidden_state.device,
                    )
                    embeddings = output.last_hidden_state[rows, lengths]
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                chunks.append(embeddings.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)


etsu_embedding_08b_memory = ModelMeta(
    loader=EtsuEncoder,
    name="Rebine/Etsu-Embedding-0.8B-Memory",
    revision="deaac7b1d4e62c12ecff3d3dbb2caa79b966c10b",
    release_date="2026-10-03",
    languages=["cmn-Hans", "eng-Latn"],
    n_parameters=752_393_024,
    n_embedding_parameters=254_279_680,
    memory_usage_mb=1505,
    max_tokens=2048,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Rebine/Etsu-Embedding-0.8B-Memory",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    model_type=["dense"],
)
