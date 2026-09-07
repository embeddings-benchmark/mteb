from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

IMAGEBIND_CITATION = """@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang and Singh, Mannat and
          Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15180--15190},
  year={2023}
}"""


class ImageBindWrapper(AbsEncoder):
    """MTEB wrapper for Meta's ImageBind model.

    ImageBind learns a single joint embedding space across image, text, and audio
    (plus depth, thermal, IMU — not used here). Output embeddings are 1024-dim
    and L2-normalized. Requires the imagebind pip package from facebookresearch.

    Audio inputs are processed in-memory via ImageBind's mel spectrogram
    pipeline without writing to disk.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from imagebind.models import imagebind_model

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.to(self.device).eval()

    def _load_text(self, texts: list[str]) -> torch.Tensor:
        from imagebind import data
        from imagebind.models.imagebind_model import ModalityType

        result = data.load_and_transform_text(texts, self.device)
        if isinstance(result, dict):
            return result[ModalityType.TEXT]
        return result

    def _load_images(self, images: list) -> torch.Tensor:
        """Transform PIL images using ImageBind's vision pipeline."""
        from PIL import Image as PILImage
        from torchvision import transforms

        # https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/imagebind/data.py#L88-L98
        transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        tensors = []
        for img in images:
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)  # noqa: PLW2901
            tensors.append(transform(img.convert("RGB")))
        return torch.stack(tensors).to(self.device)

    def _load_audio(self, audio_items: list) -> torch.Tensor:
        """Process audio arrays in-memory using ImageBind's mel spectrogram pipeline."""
        import numpy as np
        import torchaudio
        from imagebind.data import get_clip_timepoints, waveform2melspec
        from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
        from torchvision import transforms

        sample_rate = 16_000
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=2, clips_per_video=3)
        normalize = transforms.Normalize(mean=-4.268, std=9.138)

        audio_outputs = []
        for item in audio_items:
            array = np.asarray(item["array"], dtype=np.float32)
            sr = item["sampling_rate"]
            waveform = torch.from_numpy(array).unsqueeze(0)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=sample_rate
                )
            min_len = 2 * sample_rate
            if waveform.shape[1] < min_len:
                waveform = torch.nn.functional.pad(
                    waveform, (0, min_len - waveform.shape[1])
                )
            timepoints = get_clip_timepoints(
                clip_sampler, waveform.size(1) / sample_rate
            )
            clips = []
            for start, end in timepoints:
                clip = waveform[:, int(start * sample_rate) : int(end * sample_rate)]
                melspec = waveform2melspec(clip, sample_rate, 128, 204)
                clips.append(normalize(melspec).to(self.device))
            audio_outputs.append(torch.stack(clips, dim=0))

        return torch.stack(audio_outputs, dim=0)

    @torch.inference_mode()
    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        from imagebind.models.imagebind_model import ModalityType

        inputs = {}
        if batch.get("text"):
            inputs[ModalityType.TEXT] = self._load_text(batch["text"])
        if batch.get("image"):
            inputs[ModalityType.VISION] = self._load_images(batch["image"])
        if batch.get("audio"):
            inputs[ModalityType.AUDIO] = self._load_audio(batch["audio"])

        if not inputs:
            raise ValueError(
                f"No supported modality found in batch: {list(batch.keys())}"
            )

        outputs = self.model(inputs)

        embeddings = None
        for emb in outputs.values():
            embeddings = emb if embeddings is None else embeddings + emb

        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

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
        has_audio = "audio" in inputs.dataset.features

        if has_audio:
            inputs.collate_fn = AudioCollator(target_sampling_rate=16_000)

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            emb = self._encode_batch(batch)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0).float()


imagebind_huge = ModelMeta(
    loader=ImageBindWrapper,
    name="nielsr/imagebind-huge",
    revision="51fc1ff707903501e60bdb2f73dd4e8818eef099",
    n_parameters=1_200_000_000,
    n_embedding_parameters=50_331_648,
    memory_usage_mb=4578,
    max_tokens=77,
    embed_dim=1024,
    release_date="2023-05-09",
    languages=["eng-Latn"],
    license="cc-by-nc-4.0",
    open_weights=True,
    modalities=["text", "image", "audio"],
    public_training_code="https://github.com/facebookresearch/ImageBind",
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/nielsr/imagebind-huge",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    citation=IMAGEBIND_CITATION,
    extra_requirements_groups=["imagebind"],
)
