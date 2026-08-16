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


NATURELM_AUDIO_CITATION = """@misc{robinson2024naturelmaudio,
    title={NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics},
    author={Robinson, David and Miron, Marius and Hagiwara, Masato and Pietquin, Olivier},
    year={2024},
    eprint={2411.07186},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2411.07186},
}"""


class NatureLMAudioBEATsWrapper(AbsEncoder):
    """Wrapper for the BEATs audio encoder extracted from NatureLM-audio.

    This is the standalone bioacoustics-specialized encoder (avex's
    esp_aves2_naturelm_audio_v1_beats), not the full NatureLM-audio LLM
    pipeline -- pure feature extraction, no text tower.

    Zero-padding a short clip to match a longer one in the same batch
    materially distorts its embedding (cosine ~0.80 vs the unpadded
    embedding, verified directly), so same-length clips must be grouped
    and encoded separately, same issue as Dasheng's wrapper.

    BEATs' attention/embedding memory scales with sequence length, and
    some real-world field recordings (e.g. in BirdCLEF) run many minutes
    long -- one such clip triggered a 24GB single-allocation request on
    real GPU hardware, more than any single GPU here has. Clips are
    truncated to max_audio_length_seconds to keep memory bounded, same
    pattern as Qwen2AudioWrapper's max_audio_length_seconds.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        from avex import load_model
        from huggingface_hub import hf_hub_download

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_rate = 16_000
        self.max_audio_length_seconds = max_audio_length_seconds
        # avex's load_model() has no revision parameter and resolves its own
        # checkpoint by name internally, so the checkpoint is downloaded here
        # (revision-pinned) and handed to it explicitly via checkpoint_path.
        checkpoint_path = hf_hub_download(
            repo_id="EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats",
            filename="esp-aves2-naturelm-audio-v1-beats.safetensors",
            revision=revision,
        )
        self.model = load_model(
            "esp_aves2_naturelm_audio_v1_beats",
            checkpoint_path=checkpoint_path,
            return_features_only=True,
            device=self.device,
        )
        self.model.register_hooks_for_layers([0, -1])

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)

        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(
                inputs, disable=not show_progress_bar, desc="Encoding audio"
            ):
                max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
                arrays = [
                    (item["array"] if isinstance(item, dict) else item)[:max_samples]
                    for item in batch["audio"]
                ]

                groups: dict[int, list[int]] = {}
                for idx, array in enumerate(arrays):
                    groups.setdefault(len(array), []).append(idx)

                ordered: list[torch.Tensor | None] = [None] * len(arrays)
                for indices in groups.values():
                    audio_tensor = torch.stack(
                        [
                            torch.as_tensor(arrays[i], dtype=torch.float32)
                            for i in indices
                        ]
                    ).to(self.device)
                    embeddings = self.model.extract_embeddings(
                        audio_tensor, aggregation="mean"
                    )
                    for position, idx in enumerate(indices):
                        ordered[idx] = embeddings[position].cpu().to(torch.float32)

                all_embeddings.append(torch.stack(ordered))

        return torch.cat(all_embeddings, dim=0).numpy()


naturelm_audio_beats = ModelMeta(
    loader=NatureLMAudioBEATsWrapper,
    name="EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats",
    languages=["eng-Latn"],
    open_weights=True,
    revision="03f84874dacdbab04d67350fd8c0e5ee1dc04086",
    release_date="2025-08-15",
    max_tokens=float("inf"),
    n_parameters=90_717_055,
    n_embedding_parameters=0,
    memory_usage_mb=346,
    embed_dim=1536,
    license="cc-by-nc-sa-4.0",
    reference="https://huggingface.co/EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/earthspecies/NatureLM-audio",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
    citation=NATURELM_AUDIO_CITATION,
    extra_requirements_groups=["avex"],
)
