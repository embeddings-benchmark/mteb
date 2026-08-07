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


class TimesformerWrapper(AbsEncoder):
    """TimeSformer video encoder.

    TimeSformer prepends a CLS token and ``TimesformerForVideoClassification``
    reads ``last_hidden_state[:, 0]``, so we pool the same way rather than
    averaging. Checkpoints load cleanly; only the classification head is
    discarded.

    Note on the processor: ``AutoImageProcessor`` rather than
    ``AutoVideoProcessor``. ``TimesformerVideoProcessor`` only exists from
    transformers v5 onwards while ``mteb`` supports ``transformers>=4.40``.
    Every published checkpoint ships a ``preprocessor_config.json`` declaring
    ``VideoMAEImageProcessor``, which resolves on both and carries the
    checkpoint's own resize/crop/normalisation values. That matters here:
    ``TimesformerVideoProcessor`` defaults to ImageNet mean/std, so relying on
    class defaults rather than the checkpoint config would change the input.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        device: str | None = None,
        num_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.revision = revision

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, revision=revision
        )
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self.model.to(self.device)

        # TimeSformer bakes the clip length into its temporal position
        # embeddings and transformers does not interpolate it, so the config is
        # the source of truth. base is 8 frames, hr is 16.
        self.num_frames = (
            num_frames if num_frames is not None else self.model.config.num_frames
        )
        if self.num_frames != self.model.config.num_frames:
            raise ValueError(
                f"{model_name} was trained with {self.model.config.num_frames} frames "
                f"per clip, but num_frames={self.num_frames} was requested."
            )

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
        inputs.collate_fn = FramesCollator(num_frames=self.num_frames)

        embeddings = []
        for batch in tqdm(inputs, desc="Encoding", disable=not show_progress_bar):
            videos = batch["video"]
            # Explicit list-of-videos-of-frames in HWC. A list of 4D tensors trips
            # `make_batched` on the v4 slow processor, which treats the whole batch
            # as a single video and then fails inside PIL.
            frames = [[f.permute(1, 2, 0).numpy() for f in v] for v in videos]
            processed = self.processor(frames, return_tensors="pt").to(self.device)

            outputs = self.model(**processed)
            embeddings.append(outputs.last_hidden_state[:, 0].cpu())
        return torch.cat(embeddings, dim=0).numpy()


_TIMESFORMER_CITATION = """
@inproceedings{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021},
}"""

# Every published TimeSformer checkpoint is supervised-finetuned on Kinetics or
# Something-Something V2; there is no pretrained-only variant.
_K400_SUPERVISED = {
    "Kinetics400V",
    "Kinetics400VA",
    "Kinetics400ZeroShot",
    "Kinetics400VAZeroShot",
}
_SSV2_SUPERVISED = {"SomethingSomethingV2Classification"}

timesformer_base_k400 = ModelMeta(
    loader=TimesformerWrapper,
    name="facebook/timesformer-base-finetuned-k400",
    revision="8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0",
    release_date="2022-10-07",
    languages=None,
    n_parameters=121_258_752,
    n_embedding_parameters=None,
    memory_usage_mb=463,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/TimeSformer",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/facebook/timesformer-base-finetuned-k400",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_K400_SUPERVISED,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_TIMESFORMER_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

timesformer_base_ssv2 = ModelMeta(
    loader=TimesformerWrapper,
    name="facebook/timesformer-base-finetuned-ssv2",
    revision="3b045270472c79cf9c1b60189ba425e92ed7f004",
    release_date="2022-10-07",
    languages=None,
    n_parameters=121_258_752,
    n_embedding_parameters=None,
    memory_usage_mb=463,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/TimeSformer",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/facebook/timesformer-base-finetuned-ssv2",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_SSV2_SUPERVISED,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_TIMESFORMER_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

timesformer_hr_k400 = ModelMeta(
    loader=TimesformerWrapper,
    name="facebook/timesformer-hr-finetuned-k400",
    revision="188026c3022e39dac25da0111f9a015603915775",
    release_date="2022-10-07",
    languages=None,
    n_parameters=121_716_480,
    n_embedding_parameters=None,
    memory_usage_mb=464,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/facebookresearch/TimeSformer",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/facebook/timesformer-hr-finetuned-k400",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_K400_SUPERVISED,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_TIMESFORMER_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
