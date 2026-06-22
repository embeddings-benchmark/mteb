from __future__ import annotations

import sys
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
    """Byrne-Embed: an 85M SpikeWhale decoder + a 640->768 projection head producing
    unit-norm sentence embeddings. The model repo bundles its own modeling code
    (`byrne_embedder.py`, `model_v2.py`, `spike_tokenizer.py`); we download the snapshot
    and use its self-contained `ByrneEmbedder` loader.
    """

    def __init__(
        self,
        model: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from huggingface_hub import snapshot_download

        self.model_name = model
        self.device = device
        local = snapshot_download(repo_id=model, revision=revision)
        if local not in sys.path:
            sys.path.insert(0, local)
        from byrne_embedder import ByrneEmbedder

        self.encoder = ByrneEmbedder(local, device=device)

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
        for batch in inputs:
            vecs = self.encoder.encode(
                list(batch["text"]), batch_size=64, max_length=128, normalize=True
            )
            embeddings.append(vecs.to(torch.float32).cpu().numpy())
        return np.concatenate(embeddings, axis=0).astype(np.float32)


byrne_embed = ModelMeta(
    loader=ByrneEmbedModel,
    name="Quazim0t0/Byrne-Embed",
    model_type=["dense"],
    languages=["eng-Latn"],
    revision="b1dd73482beb385c295d8237a8c9c845680f8f75",
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
    framework=["PyTorch"],
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
