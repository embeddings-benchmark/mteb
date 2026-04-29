from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import FramesCollator
from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class VJepaV2Wrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        device: str | None = None,
        fps: float | None = 2.0,
        max_frames: int | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = AutoVideoProcessor.from_pretrained(
            model_name, revision=revision
        )

        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self.model.to(device)

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        inputs.collate_fn = FramesCollator(
            fps=self.fps,
            max_frames=self.max_frames,
            num_frames=self.num_frames,
        )

        embeddings = []
        for batch in tqdm(inputs, desc="Encoding", disable=not show_progress_bar):
            videos = batch["video"]
            max_frames = max(v.shape[0] for v in videos)
            padded = [
                torch.cat(
                    [v, v[-1:].expand(max_frames - v.shape[0], *v.shape[1:])], dim=0
                )
                if v.shape[0] < max_frames
                else v
                for v in videos
            ]
            processed_videos = self.processor(
                videos=padded,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**processed_videos)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0).numpy()


_JEPA_CITATION = """
@misc{assran2025vjepa2selfsupervisedvideo,
    title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
    author={Mido Assran and Adrien Bardes and David Fan and Quentin Garrido and Russell Howes and Mojtaba and Komeili and Matthew Muckley and Ammar Rizvi and Claire Roberts and Koustuv Sinha and Artem Zholus and Sergio Arnaud and Abha Gejji and Ada Martin and Francois Robert Hogan and Daniel Dugas and Piotr Bojanowski and Vasil Khalidov and Patrick Labatut and Francisco Massa and Marc Szafraniec and Kapil Krishnakumar and Yong Li and Xiaodong Ma and Sarath Chandar and Franziska Meier and Yann LeCun and Michael Rabbat and Nicolas Ballas},
    year={2025},
    eprint={2506.09985},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2506.09985},
}"""

vjepa2_vitl_fpc64_256 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitl-fpc64-256",
    revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
    release_date="2025-06-11",
    languages=None,
    n_parameters=325_971_328,
    n_embedding_parameters=None,
    memory_usage_mb=1243,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc64-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vith_fpc64_256 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vith-fpc64-256",
    revision="b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5",
    release_date="2025-06-11",
    languages=None,
    n_parameters=653930880,
    n_embedding_parameters=None,
    memory_usage_mb=2495,
    max_tokens=None,
    embed_dim=1280,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vith-fpc64-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitg_fpc64_256 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitg-fpc64-256",
    revision="875c192b7b704b87d1e1d99345769632dd5f739a",
    release_date="2025-06-11",
    languages=None,
    n_parameters=1034555264,
    n_embedding_parameters=None,
    memory_usage_mb=3947,
    max_tokens=None,
    embed_dim=1408,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-256",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitg_fpc64_384 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitg-fpc64-384",
    revision="12ca91694b230e0d4b5b0078af6f4ae1d51e933d",
    release_date="2025-06-11",
    languages=None,
    n_parameters=1034555264,
    n_embedding_parameters=None,
    memory_usage_mb=3947,
    max_tokens=None,
    embed_dim=1408,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-384",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitg_fpc64_384_ssv2 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitg-fpc64-384-ssv2",
    revision="9f5fd615cb6f79065a28edcf1cc3ef25010dddfa",
    release_date="2025-06-13",
    languages=None,
    n_parameters=1128049454,
    n_embedding_parameters=None,
    memory_usage_mb=4303,
    max_tokens=None,
    embed_dim=1408,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc64-384-ssv2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        "SomethingSomethingV2Classification",
    ),
    adapted_from="facebook/vjepa2-vitg-fpc64-384",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitl_fpc16_256_ssv2 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitl-fpc16-256-ssv2",
    revision="4aa02df83918538fc21cfaf576382fa20e489a80",
    release_date="2025-06-13",
    languages=None,
    n_parameters=375485998,
    n_embedding_parameters=None,
    memory_usage_mb=1432,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        "SomethingSomethingV2Classification",
    ),
    adapted_from="facebook/vjepa2-vitl-fpc64-256",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitg_fpc32_384_diving48 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitg-fpc32-384-diving48",
    revision="0b48243375319bd8e03e3cd5560d957095429189",
    release_date="2025-06-13",
    languages=None,
    n_parameters=1127871920,
    n_embedding_parameters=None,
    memory_usage_mb=4302,
    max_tokens=None,
    embed_dim=1408,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitg-fpc32-384-diving48",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # "Diving48Classification",
    ),
    adapted_from="facebook/vjepa2-vitg-fpc64-384",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

vjepa2_vitl_fpc32_256_diving48 = ModelMeta(
    loader=VJepaV2Wrapper,
    name="facebook/vjepa2-vitl-fpc32-256-diving48",
    revision="71ae2a8b1ff5a297aeeaae9b5e64c7a2e5e6a633",
    release_date="2025-06-13",
    languages=None,
    n_parameters=375356848,
    n_embedding_parameters=None,
    memory_usage_mb=1432,
    max_tokens=None,
    embed_dim=1024,
    license="mit",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["Transformers", "safetensors"],
    reference="https://huggingface.co/facebook/vjepa2-vitl-fpc32-256-diving48",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(
        # "Diving48Classification",
    ),
    adapted_from="facebook/vjepa2-vitl-fpc64-256",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_JEPA_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
