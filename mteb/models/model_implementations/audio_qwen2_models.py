from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from mteb._create_dataloaders import AudioCollator
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class Qwen2AudioWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_audio_length_seconds: float = 30.0,
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        self.model_name = model_name
        self.device = device
        self.max_audio_length_seconds = max_audio_length_seconds

        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate

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
        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            audio_arrays = []
            texts = []
            audio_list = batch.get("audio", [])
            text_list = batch.get("text", [])
            batch_size = max(len(audio_list), len(text_list))

            for i in range(batch_size):
                cur_text = ""
                audio_row = audio_list[i] if i < len(audio_list) else None
                if audio_row is not None:
                    array = AudioCollator.resample_audio(
                        {"audio": audio_row},
                        self.sampling_rate,
                    )
                    cur_text += "<|audio_bos|><|AUDIO|><|audio_eos|>"
                    audio_arrays.append(array)

                text_row = text_list[i] if i < len(text_list) else None
                if text_row is not None:
                    cur_text += text_row
                texts.append(cur_text)

            processor_inputs = self.processor(
                text=texts,
                audio=audio_arrays if len(audio_arrays) > 0 else None,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(self.max_audio_length_seconds * self.sampling_rate),
            )

            processor_inputs = {
                k: v.to(self.device) for k, v in processor_inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(
                    **processor_inputs,
                    output_hidden_states=True,
                )

                hidden = outputs.hidden_states[-1]
                mask = processor_inputs["attention_mask"]

                # last non-pad index per item
                last_idx = mask.sum(dim=1) - 1
                last_idx = last_idx.clamp(min=0)

                # gather last-token embeddings
                batch_indices = torch.arange(hidden.size(0), device=self.device)
                embeddings = hidden[batch_indices, last_idx]

                all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()


qwen2_audio_meta = ModelMeta(
    loader=Qwen2AudioWrapper,
    name="Qwen/Qwen2-Audio-7B",
    languages=["eng-Latn"],
    open_weights=True,
    revision="dd84470756e6277a71d4d7188773a43cde92696e",
    release_date="2024-08-09",
    max_tokens=131_072,
    n_parameters=7_000_000_000,
    memory_usage_mb=None,
    embed_dim=1280,
    license="mit",
    reference="https://huggingface.co/Qwen/Qwen2-Audio-7B",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
    modalities=["audio", "text"],
    citation="""
@misc{chu2024qwen2audiotechnicalreport,
      title={Qwen2-Audio Technical Report},
      author={Yunfei Chu and Jin Xu and Qian Yang and Haojie Wei and Xipin Wei and Zhifang Guo and Yichong Leng and Yuanjun Lv and Jinzheng He and Junyang Lin and Chang Zhou and Jingren Zhou},
      year={2024},
      eprint={2407.10759},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2407.10759},
}
""",
)
