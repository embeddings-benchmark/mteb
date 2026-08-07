from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


def _load_checkpoint_tensors(
    model_name: str, revision: str | None
) -> dict[str, torch.Tensor]:
    """Read the raw checkpoint tensors for a Hub model.

    Called after ``from_pretrained``, so the file is already in the local
    HuggingFace cache and this does not hit the network again.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(model_name, "model.safetensors", revision=revision)
    except EntryNotFoundError:
        path = hf_hub_download(model_name, "pytorch_model.bin", revision=revision)
        return torch.load(path, map_location="cpu", weights_only=True)

    from safetensors.torch import load_file

    return load_file(path)


class VideoMAEWrapper(AbsEncoder):
    """VideoMAE encoder.

    VideoMAE has no CLS token and trains with ``use_mean_pooling=True``, so
    ``VideoMAEForVideoClassification`` averages ``last_hidden_state`` over the
    token axis before its head. We pool the same way.

    Two weight fixups run after loading, both of which change the output. See
    ``_restore_qkv_bias`` and ``_load_fc_norm``.

    Note on the processor: ``AutoImageProcessor`` rather than
    ``AutoVideoProcessor``. ``VideoMAEVideoProcessor`` only exists from
    transformers v5 onwards while ``mteb`` supports ``transformers>=4.40``.
    Every published checkpoint ships a ``preprocessor_config.json`` declaring
    ``VideoMAEImageProcessor``, which resolves on both and carries the
    checkpoint's own resize/crop/normalisation values.
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

        state_dict = _load_checkpoint_tensors(model_name, revision)
        self._load_fc_norm(state_dict)

        self.model.to(self.device)
        if self.fc_norm is not None:
            self.fc_norm.to(self.device)

        # VideoMAE bakes the clip length into its temporal position embeddings
        # and transformers does not interpolate it, so the config is the source
        # of truth rather than a per-ModelMeta constant.
        self.num_frames = (
            num_frames if num_frames is not None else self.model.config.num_frames
        )
        if self.num_frames != self.model.config.num_frames:
            raise ValueError(
                f"{model_name} was trained with {self.model.config.num_frames} frames "
                f"per clip, but num_frames={self.num_frames} was requested."
            )

    def _load_fc_norm(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the final LayerNorm when it lives on the classification head.

        ``VideoMAEModel.layernorm`` exists only when ``use_mean_pooling`` is
        False, which is the case for the self-supervised checkpoints. The
        finetuned ones set it True, moving that LayerNorm to ``fc_norm`` on the
        classification head, which ``AutoModel`` discards. Without this the two
        groups would produce embeddings through different pipelines.
        """
        self.fc_norm: torch.nn.LayerNorm | None = None
        if not getattr(self.model.config, "use_mean_pooling", False):
            return  # base model keeps its own layernorm

        if "fc_norm.weight" not in state_dict:
            logger.warning(
                "%s sets use_mean_pooling=True but has no fc_norm in the "
                "checkpoint; embeddings will be un-normalised.",
                self.model_name,
            )
            return

        fc_norm = torch.nn.LayerNorm(self.model.config.hidden_size)
        fc_norm.load_state_dict(
            {
                "weight": state_dict["fc_norm.weight"],
                "bias": state_dict["fc_norm.bias"],
            }
        )
        fc_norm.eval()
        self.fc_norm = fc_norm

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
            padded = [
                torch.cat(
                    [v, v[-1:].expand(self.num_frames - v.shape[0], *v.shape[1:])],
                    dim=0,
                )
                if v.shape[0] < self.num_frames
                else v[: self.num_frames]
                for v in videos
            ]
            processed = self.processor(padded, return_tensors="pt").to(self.device)

            outputs = self.model(**processed)
            pooled = outputs.last_hidden_state.mean(dim=1)
            if self.fc_norm is not None:
                pooled = self.fc_norm(pooled)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0).numpy()


_VIDEOMAE_CITATION = """
@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}"""

# Self-supervised pre-training uses Kinetics-400 videos without labels.
# Declared so the leaderboard flags the overlap even though no label was seen.
_K400 = {
    "Kinetics400V",
    "Kinetics400VA",
    "Kinetics400ZeroShot",
    "Kinetics400VAZeroShot",
}

videomae_base = ModelMeta(
    loader=VideoMAEWrapper,
    name="MCG-NJU/videomae-base",
    revision="dc740ceda42fce44faed2ea03c6d447db72f6af9",
    release_date="2022-08-03",
    languages=None,
    n_parameters=86_236_416,
    n_embedding_parameters=None,
    memory_usage_mb=329,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/MCG-NJU/VideoMAE",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/MCG-NJU/videomae-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_K400,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=["transformers-v4"],
)

videomae_large = ModelMeta(
    loader=VideoMAEWrapper,
    name="MCG-NJU/videomae-large",
    revision="12da269a02d3e1fbbb7011e610e591fea8061dca",
    release_date="2022-08-02",
    languages=None,
    n_parameters=303_885_312,
    n_embedding_parameters=None,
    memory_usage_mb=1159,
    max_tokens=None,
    embed_dim=1024,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/MCG-NJU/VideoMAE",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/MCG-NJU/videomae-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_K400,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=["transformers-v4"],
)

videomae_base_finetuned_kinetics = ModelMeta(
    loader=VideoMAEWrapper,
    name="MCG-NJU/videomae-base-finetuned-kinetics",
    revision="488eb9a0565f257b32866000305c8178965eb9f6",
    release_date="2022-07-08",
    languages=None,
    n_parameters=86_234_880,
    n_embedding_parameters=None,
    memory_usage_mb=329,
    max_tokens=None,
    embed_dim=768,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code="https://github.com/MCG-NJU/VideoMAE",
    public_training_data=None,
    framework=["Transformers", "PyTorch"],
    reference="https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=_K400,
    adapted_from="MCG-NJU/videomae-base",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=["transformers-v4"],
)
