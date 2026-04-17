from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
)

from mteb._create_dataloaders import AudioCollator, VideoCollator
from mteb._requires_package import (
    requires_audio_dependencies,
    requires_image_dependencies,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


class QwenOmniWrapper(AbsEncoder):
    """Wrapper for Qwen Omni models supporting audio, images, and video. Last token pooling is used to get the embedding."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        max_audio_length_seconds: int = 300,
        num_frames: int = 16,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_audio_dependencies()
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length_seconds = max_audio_length_seconds
        self.num_frames = num_frames

        if "2.5" in model_name:
            from transformers import Qwen2_5OmniThinkerForConditionalGeneration

            model_class = Qwen2_5OmniThinkerForConditionalGeneration
        elif "3" in model_name:
            from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

            model_class = Qwen3OmniMoeThinkerForConditionalGeneration

        self.model = model_class.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.max_samples = int(self.max_audio_length_seconds * self.sampling_rate)

    @staticmethod
    def _resize_video(video: torch.Tensor) -> torch.Tensor:
        """Pre-resize video frames via smart_resize to match the model's expected pixel range.

        Replicates the smart_resize step from qwen_omni_utils fetch_video which
        constrains frames to 100352-602112 pixels before the processor applies
        its own resize.
        """
        from torchvision.transforms.functional import InterpolationMode, resize
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize,
        )

        patch_size = 14
        merge_size = 2
        factor = patch_size * merge_size  # 28
        min_pixels = 128 * factor * factor  # 100352
        max_pixels = 768 * factor * factor  # 602112

        _, _, h, w = video.shape
        new_h, new_w = smart_resize(
            h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        if new_h != h or new_w != w:
            video = resize(
                video,
                [new_h, new_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return video

    @staticmethod
    def _build_messages(
        batch_texts: list[str],
        batch_images: list[Any],
        batch_audio: list[Any],
        batch_video: list[Any],
    ) -> list[list[dict[str, Any]]]:
        messages = []
        batch_size = max(
            len(batch_texts), len(batch_images), len(batch_audio), len(batch_video)
        )
        for i in range(batch_size):
            text_content = batch_texts[i] if i < len(batch_texts) else ""
            image_content = batch_images[i] if i < len(batch_images) else None
            audio_content = batch_audio[i] if i < len(batch_audio) else None
            video_content = batch_video[i] if i < len(batch_video) else None

            content = []
            if video_content is not None:
                content.append({"type": "video", "video": video_content})
            if audio_content is not None:
                content.append({"type": "audio", "audio": audio_content})
            if image_content is not None:
                content.append(
                    {
                        "type": "image",
                        "image": image_content,
                    }
                )
            content.append({"type": "text", "text": text_content})
            messages.append(
                [
                    # qwen2.5 audio output mode only works when using default system prompt
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                            }
                        ],
                    },
                    {"role": "user", "content": content},
                ]
            )
        return messages

    @torch.no_grad()
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
        if "video" in inputs.dataset.features or "audio" in inputs.dataset.features:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                max_frames=self.num_frames,
                max_samples=self.max_samples,
            )

        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, desc="Encoding"):
            batch_texts = batch.get("text", [])
            batch_images = batch.get("image", [])
            raw_audio = batch.get("audio", [])
            raw_video = batch.get("video", [])

            batch_audio = []
            for audio_row in raw_audio:
                if audio_row is None:
                    batch_audio.append(None)
                elif isinstance(audio_row, dict) and "array" in audio_row:
                    batch_audio.append(audio_row["array"])
                else:
                    array = AudioCollator.resample_audio(
                        {"audio": audio_row}, self.sampling_rate, self.max_samples
                    )
                    batch_audio.append(array)

            batch_video = []
            for video_row in raw_video:
                if video_row is None:
                    batch_video.append(None)
                elif isinstance(video_row, torch.Tensor):
                    batch_video.append(self._resize_video(video_row))
                else:
                    batch_video.append(video_row)

            messages = self._build_messages(
                batch_texts=batch_texts,
                batch_images=batch_images,
                batch_audio=batch_audio,
                batch_video=batch_video,
            )

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]


            audio_inputs = [a for a in batch_audio if a is not None] or None
            video_inputs = [v for v in batch_video if v is not None] or None
            image_inputs = [img for img in batch_images if img is not None] or None

            model_inputs = self.processor(
                text=texts,
                audio=audio_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                use_audio_in_video=False,
            ).to(self.device).to(self.model.dtype)

            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
            embeddings = outputs.hidden_states[-1][
                :, -1
            ]  # select last hidden state ([-1]) and last token position ([:, -1]).
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).float()


qwen25_omni_7b = ModelMeta(
    loader=QwenOmniWrapper,
    name="Qwen/Qwen2.5-Omni-7B",
    revision="ae9e1690543ffd5c0221dc27f79834d0294cba00",
    release_date="2025-03-22",
    languages=["eng-Latn"],
    n_parameters=10_732_225_440,
    memory_usage_mb=21327.0,
    max_tokens=32768,
    embed_dim=3584,
    n_embedding_parameters=544_997_376,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen2.5-Omni-7B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=[
        "text",
        "image",
        "audio",
        "video",
    ],
    model_type=["dense"],
    citation="""
@misc{xu2025qwen25omnitechnicalreport,
    title={Qwen2.5-Omni Technical Report},
    author={Jin Xu and Zhifang Guo and Jinzheng He and Hangrui Hu and Ting He and Shuai Bai and Keqin Chen and Jialin Wang and Yang Fan and Kai Dang and Bin Zhang and Xiong Wang and Yunfei Chu and Junyang Lin},
    year={2025},
    eprint={2503.20215},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.20215},
}""",
)

qwen25_omni_3b = ModelMeta(
    loader=QwenOmniWrapper,
    name="Qwen/Qwen2.5-Omni-3B",
    revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
    release_date="2025-04-30",
    languages=["eng-Latn"],
    n_parameters=5_537_120_672,
    memory_usage_mb=11418.0,
    max_tokens=32768,
    embed_dim=2048,
    n_embedding_parameters=311_164_928,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen2.5-Omni-3B",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=[
        "text",
        "image",
        "audio",
        "video",
    ],
    model_type=["dense"],
    citation="""
@misc{xu2025qwen25omnitechnicalreport,
    title={Qwen2.5-Omni Technical Report},
    author={Jin Xu and Zhifang Guo and Jinzheng He and Hangrui Hu and Ting He and Shuai Bai and Keqin Chen and Jialin Wang and Yang Fan and Kai Dang and Bin Zhang and Xiong Wang and Yunfei Chu and Junyang Lin},
    year={2025},
    eprint={2503.20215},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2503.20215},
}""",
)


qwen3_omni_30b_a3b_instruct = ModelMeta(
    loader=QwenOmniWrapper,
    name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    revision="26291f793822fb6be9555850f06dfe95f2d7e695",
    release_date="2025-09-20",
    languages=["eng-Latn"],
    n_parameters=35259818545,
    memory_usage_mb=67253.0,
    max_tokens=151_643,
    embed_dim=2048,
    n_embedding_parameters=155_713_536,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=[
        "text",
        "image",
        "audio",
        "video",
    ],
    model_type=["dense"],
    citation="""
@misc{xu2025qwen3omnitechnicalreport,
    title={Qwen3-Omni Technical Report},
    author={Jin Xu and Zhifang Guo and Hangrui Hu and Yunfei Chu and Xiong Wang and Jinzheng He and Yuxuan Wang and Xian Shi and Ting He and Xinfa Zhu and Yuanjun Lv and Yongqi Wang and Dake Guo and He Wang and Linhan Ma and Pei Zhang and Xinyu Zhang and Hongkun Hao and Zishan Guo and Baosong Yang and Bin Zhang and Ziyang Ma and Xipin Wei and Shuai Bai and Keqin Chen and Xuejing Liu and Peng Wang and Mingkun Yang and Dayiheng Liu and Xingzhang Ren and Bo Zheng and Rui Men and Fan Zhou and Bowen Yu and Jianxin Yang and Le Yu and Jingren Zhou and Junyang Lin},
    year={2025},
    eprint={2509.17765},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.17765},
}""",
)

qwen3_omni_30b_a3b_thinking = ModelMeta(
    loader=QwenOmniWrapper,
    name="Qwen/Qwen3-Omni-30B-A3B-Thinking",
    revision="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b",
    release_date="2025-09-15",
    languages=["eng-Latn"],
    n_parameters=31719205488,
    memory_usage_mb=60500.0,
    max_tokens=151_643,
    embed_dim=2048,
    n_embedding_parameters=155_713_536,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=[
        "text",
        "image",
        "audio",
        "video",
    ],
    model_type=["dense"],
    citation="""
@misc{xu2025qwen3omnitechnicalreport,
    title={Qwen3-Omni Technical Report},
    author={Jin Xu and Zhifang Guo and Hangrui Hu and Yunfei Chu and Xiong Wang and Jinzheng He and Yuxuan Wang and Xian Shi and Ting He and Xinfa Zhu and Yuanjun Lv and Yongqi Wang and Dake Guo and He Wang and Linhan Ma and Pei Zhang and Xinyu Zhang and Hongkun Hao and Zishan Guo and Baosong Yang and Bin Zhang and Ziyang Ma and Xipin Wei and Shuai Bai and Keqin Chen and Xuejing Liu and Peng Wang and Mingkun Yang and Dayiheng Liu and Xingzhang Ren and Bo Zheng and Rui Men and Fan Zhou and Bowen Yu and Jianxin Yang and Le Yu and Jingren Zhou and Junyang Lin},
    year={2025},
    eprint={2509.17765},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.17765},
}""",
)

qwen3_omni_30b_a3b_captioner = ModelMeta(
    loader=QwenOmniWrapper,
    name="Qwen/Qwen3-Omni-30B-A3B-Captioner",
    revision="a2bd106cbf527db5676e79662674da22b0545ec0",
    release_date="2025-09-15",
    languages=["eng-Latn"],
    n_parameters=31719205488,
    memory_usage_mb=60500.0,
    max_tokens=151_643,
    embed_dim=2048,
    n_embedding_parameters=155_713_536,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=[
        "text",
        "image",
        "audio",
        "video",
    ],
    model_type=["dense"],
    citation="""
@misc{xu2025qwen3omnitechnicalreport,
    title={Qwen3-Omni Technical Report},
    author={Jin Xu and Zhifang Guo and Hangrui Hu and Yunfei Chu and Xiong Wang and Jinzheng He and Yuxuan Wang and Xian Shi and Ting He and Xinfa Zhu and Yuanjun Lv and Yongqi Wang and Dake Guo and He Wang and Linhan Ma and Pei Zhang and Xinyu Zhang and Hongkun Hao and Zishan Guo and Baosong Yang and Bin Zhang and Ziyang Ma and Xipin Wei and Shuai Bai and Keqin Chen and Xuejing Liu and Peng Wang and Mingkun Yang and Dayiheng Liu and Xingzhang Ren and Bo Zheng and Rui Men and Fan Zhou and Bowen Yu and Jianxin Yang and Le Yu and Jingren Zhou and Junyang Lin},
    year={2025},
    eprint={2509.17765},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.17765},
}""",
)
