from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class ByrneEmbedModel(AbsEncoder):
    """Byrne-Embed: an 85M SpikeWhale decoder + a fused 640->768 projection head that
    produces unit-norm sentence embeddings. The model is a custom-code `transformers`
    model loaded with `trust_remote_code=True` (modeling code lives in the model repo).
    """

    def __init__(
        self,
        model: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 128,
        **kwargs: Any,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(model, revision=revision, trust_remote_code=True)
            .to(device)
            .eval()
        )

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
        embeddings = []
        with torch.no_grad():
            for batch in inputs:
                enc = self.tokenizer(
                    list(batch["text"]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                emb = self.model(**enc).last_hidden_state  # (B, 768), L2-normalized
                embeddings.append(emb.to(torch.float32).cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)


byrne_embed = ModelMeta(
    loader=ByrneEmbedModel,
    name="Quazim0t0/Byrne-Embed",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="af905dd5dfa2b42a07f8d953dd25171709257e8d",
    release_date="2026-06-20",
    modalities=["text"],
    n_parameters=85_700_000,
    memory_usage_mb=327,
    max_tokens=128,
    embed_dim=768,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/Quazim0t0/Byrne-Embed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    citation="""@misc{byrne2026byrneembed,
  title        = {Byrne-Embed: A Compact 85M Sentence-Embedding Model},
  author       = {Byrne, Dean},
  year         = {2026},
  howpublished = {\\url{https://huggingface.co/Quazim0t0/Byrne-Embed}},
}""",
)
