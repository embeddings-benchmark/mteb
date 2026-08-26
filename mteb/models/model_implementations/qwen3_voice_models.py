from __future__ import annotations

import logging
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
    from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)


class Qwen3VoiceEmbeddingWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        from transformers import AutoFeatureExtractor, AutoModel

        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        self.sampling_rate = self.feature_extractor.sampling_rate

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
        inputs.collate_fn = AudioCollator(
            target_sampling_rate=self.sampling_rate, max_samples=max_samples
        )
        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            audio_arrays = [audio["array"] for audio in batch["audio"]]

            feature_inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            feature_inputs = {k: v.to(self.device) for k, v in feature_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**feature_inputs)
                embeddings = outputs.last_hidden_state
                all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()

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
        if "audio" not in inputs.dataset.features:
            raise ValueError("Qwen3VoiceEmbeddingWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


qwen3_voice_embedding_1_7b = ModelMeta(
    loader=Qwen3VoiceEmbeddingWrapper,
    name="marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="7577f61c42737fc8064bba773e2a18602df92803",
    release_date="2026-02-09",
    max_tokens=float("inf"),
    n_parameters=12_001_088,
    n_embedding_parameters=0,
    memory_usage_mb=23,
    embed_dim=2048,
    license="apache-2.0",
    reference="https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
    citation="""
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
""",
    extra_requirements_groups=["fusion-embedding"],
)


qwen3_voice_embedding_0_6b = ModelMeta(
    loader=Qwen3VoiceEmbeddingWrapper,
    name="marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="93f9dff7198816748fd8263b03298b351ca36cd8",
    release_date="2026-02-09",
    max_tokens=float("inf"),
    n_parameters=8_854_336,
    n_embedding_parameters=0,
    memory_usage_mb=17,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
    citation="""
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
""",
    extra_requirements_groups=["fusion-embedding"],
)
