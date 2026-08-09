from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class DashengAudioWrapper(AbsEncoder):
    """Wrapper for Dasheng masked-autoencoder audio encoders.

    DashengModel returns a SequenceClassifierOutput whose ``logits`` field is
    ``sigmoid(mean(hidden_states))`` -- the pooling head applies a sigmoid when
    ``encoder.pooling == "mean"``. That squashes every dimension into (0, 1) and
    destroys the geometry cosine similarity relies on, so the embedding is taken
    from the mean-pooled ``hidden_states`` instead.

    Note also that ``hidden_states`` here is a raw (B, T, D) tensor rather than the
    tuple-of-layers the field normally holds, so it must not be indexed as one.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True, **kwargs
        ).to(self.device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        self.sampling_rate = self.feature_extractor.sampling_rate

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
        return self.get_audio_embeddings(inputs, **kwargs)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[BatchedInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                inputs, disable=not show_progress_bar, desc="Encoding audio"
            ):
                arrays = [
                    item["array"] if isinstance(item, dict) else item
                    for item in batch["audio"]
                ]
                # The feature extractor pads to the longest clip and emits no
                # attention mask, and attention is global over the padded
                # sequence. A 1s clip batched with a 5s clip therefore embeds
                # differently than alone (cosine 0.41 in testing), while
                # equal-length clips batch exactly (max abs diff 0.0). So group
                # by length and encode each group separately.
                groups: dict[int, list[int]] = {}
                for idx, array in enumerate(arrays):
                    groups.setdefault(len(array), []).append(idx)

                ordered: list[torch.Tensor | None] = [None] * len(arrays)
                for indices in groups.values():
                    features = self.feature_extractor(
                        [arrays[i] for i in indices],
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                    )
                    features = {k: v.to(self.device) for k, v in features.items()}
                    outputs = self.model(**features)
                    # See class docstring: mean-pooled hidden states, not `logits`.
                    embeddings = (
                        outputs.hidden_states.mean(dim=1).cpu().to(torch.float32)
                    )
                    for position, idx in enumerate(indices):
                        ordered[idx] = embeddings[position]

                all_embeddings.append(torch.stack(ordered))

        return torch.cat(all_embeddings, dim=0)
