from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class PenguinEncoderModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = device
        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        config.rope_type = "linear"  # for
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            revision=revision,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )

    @torch.inference_mode
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = False,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        all_image_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar, desc="Image Encoding"):
            inputs = self.processor(images=batch["image"], merge_size=1)
            inputs = {k: torch.tensor(v) for k, v in inputs.items()}

            pixel_values = inputs.pop("pixel_values").to(
                self.device, dtype=torch.bfloat16
            )
            other_inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self.model(pixel_values=pixel_values, **other_inputs)
            batch_embeddings = image_features.mean(dim=0, keepdim=True)
            all_image_embeddings.append(batch_embeddings.cpu().float())

        return torch.cat(all_image_embeddings, dim=0)


PENGUIN_CITATION = """@misc{zhang2026penguinvlexploringefficiencylimits,
    title={Penguin-VL: Exploring the Efficiency Limits of VLM with LLM-based Vision Encoders},
    author={Boqiang Zhang and Lei Ke and Ruihan Yang and Qi Gao and Tianyuan Qu and Rossell Chen and Dong Yu and Leoweiliang},
    year={2026},
    eprint={2603.06569},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.06569},
}"""


penguin_encoder = ModelMeta(
    loader=PenguinEncoderModel,
    name="tencent/Penguin-Encoder",
    revision="62327d397d6034c6c7c91682c72f877f9fbf072d",
    release_date="2026-03-05",
    languages=["eng-Latn"],
    n_parameters=441_070_592,
    n_embedding_parameters=155_582_464,
    memory_usage_mb=841,
    max_tokens=40960,
    embed_dim=1024,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/tencent/Penguin-Encoder",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from="Qwen/Qwen3-0.6B",
    superseded_by=None,
    modalities=["image"],
    model_type=["dense"],
    citation=PENGUIN_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=[
        "mctct",  # have rope init error with transformers v5
        "flash_attention",
    ],
)
