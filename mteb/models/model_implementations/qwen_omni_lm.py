from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
)

from mteb._create_dataloaders import AudioCollator, VideoCollator
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
        fps: float | None = 2.0,
        max_frames: int | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length_seconds = max_audio_length_seconds
        self.fps = fps
        self.max_frames = max_frames
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
    def _build_messages(
        batch: BatchedInput,
    ) -> list[list[dict[str, Any]]]:
        """Build chat messages from a batch for apply_chat_template."""
        texts = batch.get("text", [])
        images = batch.get("image", [])
        audios = batch.get("audio", [])
        videos = batch.get("video", [])
        batch_size = max(len(texts), len(images), len(audios), len(videos))

        messages = []
        for i in range(batch_size):
            content: list[dict[str, Any]] = []
            if i < len(videos) and videos[i] is not None:
                content.append({"type": "video", "video": "placeholder"})
            if i < len(audios) and audios[i] is not None:
                content.append({"type": "audio", "audio": "placeholder"})
            if i < len(images) and images[i] is not None:
                content.append({"type": "image", "image": "placeholder"})
            content.append({"type": "text", "text": texts[i] if i < len(texts) else ""})
            messages.append(
                [
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
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features
        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                max_samples=self.max_samples,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_samples,
            )

        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, desc="Encoding"):
            messages = self._build_messages(batch)

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]

            videos = batch.get("video")
            images = batch.get("image")
            audios = batch.get("audio")
            if audios:
                audios = [
                    a["array"] if isinstance(a, dict) and "array" in a else a
                    for a in audios
                ]

            model_inputs = (
                self.processor(
                    text=texts,
                    audio=audios or None,
                    images=images or None,
                    videos=videos or None,
                    padding=True,
                    return_tensors="pt",
                    videos_kwargs={
                        "do_sample_frames": False,
                        "use_audio_in_video": False,
                    },
                    audio_kwargs={"max_length": self.max_samples},
                )
                .to(self.device)
                .to(self.model.dtype)
            )

            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
            embeddings = outputs.hidden_states[-1][:, -1]
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
    extra_requirements_groups=["qwen_omni_utils"],
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
    extra_requirements_groups=["qwen_omni_utils"],
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
    extra_requirements_groups=["qwen_omni_utils"],
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
    extra_requirements_groups=["qwen_omni_utils"],
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
    extra_requirements_groups=["qwen_omni_utils"],
)
