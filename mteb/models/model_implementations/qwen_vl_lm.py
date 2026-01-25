from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from mteb._requires_package import (
    requires_audio_dependencies,
    requires_image_dependencies,
    requires_package,
)
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType


class QwenOmniWrapper(AbsEncoder):
    """Wrapper for Qwen Omni models supporting audio and images."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        model_class,
        device: str | None = None,
        max_audio_length_seconds: int = 10,
        **kwargs: Any,
    ) -> None:
        requires_image_dependencies()
        requires_audio_dependencies()
        requires_package(
            self, "qwen_omni_utils", model_name, "pip install mteb[qwen_omni_utils]"
        )
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.max_audio_length_seconds = max_audio_length_seconds

        self.model = model_class.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def _prepare_audio(self, audio_row: dict[str, Any]) -> torch.Tensor:
        import torchaudio

        array = torch.tensor(audio_row["array"], dtype=torch.float32)
        sr = audio_row.get("sampling_rate", self.sampling_rate)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sampling_rate
            )
            array = resampler(array)
        max_samples = int(self.max_audio_length_seconds * self.sampling_rate)
        if len(array) > max_samples:
            array = array[:max_samples]
        return array

    def _build_messages(
        self,
        batch_texts: list[str],
        batch_images: list[Any],
        batch_audio: list[Any],
    ) -> list[list[dict[str, Any]]]:
        messages = []
        batch_size = max(len(batch_texts), len(batch_images), len(batch_audio))
        for i in range(batch_size):
            text_content = batch_texts[i] if i < len(batch_texts) else ""
            image_content = batch_images[i] if i < len(batch_images) else None
            audio_content = batch_audio[i] if i < len(batch_audio) else None

            content = []
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
        from qwen_omni_utils import process_mm_info

        all_embeddings: list[torch.Tensor] = []

        for batch in tqdm(inputs, desc="Encoding"):
            batch_texts = batch.get("text", [])
            batch_images = batch.get("image", [])
            raw_audio = batch.get("audio", [])

            batch_audio = []
            for audio_row in raw_audio:
                if audio_row is None:
                    batch_audio.append(None)
                    continue
                array = self._prepare_audio(audio_row)
                batch_audio.append(array.numpy())

            messages = self._build_messages(
                batch_texts=batch_texts,
                batch_images=batch_images,
                batch_audio=batch_audio,
            )

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                for msg in messages
            ]

            audio_inputs, image_inputs, video_inputs = process_mm_info(
                messages, use_audio_in_video=False
            )

            model_inputs = self.processor(
                text=texts,
                audio=audio_inputs,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                use_audio_in_video=False,
            ).to(self.device)

            outputs = self.model(
                **model_inputs, output_hidden_states=True, return_dict=True
            )
            embeddings = outputs.hidden_states[-1][:, -1]
            embeddings = embeddings.to(torch.float32)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


qwen25_omni_7b = ModelMeta(
    loader=QwenOmniWrapper,
    loader_kwargs=dict(
        model_class=Qwen2_5OmniThinkerForConditionalGeneration,
    ),
    name="Qwen/Qwen2.5-Omni-7B",
    revision="ae9e1690543ffd5c0221dc27f79834d0294cba00",
    release_date="2025-03-22",
    languages=["eng-Latn"],
    n_parameters=10_732_225_440,
    memory_usage_mb=21327.0,
    max_tokens=32768,
    embed_dim=3584,
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
    loader_kwargs=dict(
        model_class=Qwen2_5OmniThinkerForConditionalGeneration,
    ),
    name="Qwen/Qwen2.5-Omni-3B",
    revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
    release_date="2025-04-30",
    languages=["eng-Latn"],
    n_parameters=5_537_120_672,
    memory_usage_mb=11418.0,
    max_tokens=32768,
    embed_dim=2048,
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
    loader_kwargs=dict(
        model_class=Qwen3OmniMoeThinkerForConditionalGeneration,
    ),
    name="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    revision="26291f793822fb6be9555850f06dfe95f2d7e695",
    release_date="2025-09-20",
    languages=["eng-Latn"],
    n_parameters=35259818545,
    memory_usage_mb=67253.0,
    max_tokens=151_643,
    embed_dim=2048,
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
    loader_kwargs=dict(
        model_class=Qwen3OmniMoeThinkerForConditionalGeneration,
    ),
    name="Qwen/Qwen3-Omni-30B-A3B-Thinking",
    revision="2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b",
    release_date="2025-09-15",
    languages=["eng-Latn"],
    n_parameters=31719205488,
    memory_usage_mb=60500.0,
    max_tokens=151_643,
    embed_dim=2048,
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
    loader_kwargs=dict(
        model_class=Qwen3OmniMoeThinkerForConditionalGeneration,
    ),
    name="Qwen/Qwen3-Omni-30B-A3B-Captioner",
    revision="a2bd106cbf527db5676e79662674da22b0545ec0",
    release_date="2025-09-15",
    languages=["eng-Latn"],
    n_parameters=31719205488,
    memory_usage_mb=60500.0,
    max_tokens=151_643,
    embed_dim=2048,
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
