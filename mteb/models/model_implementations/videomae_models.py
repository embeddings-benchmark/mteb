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
    """Read raw checkpoint tensors. The file is already in the local HF cache."""
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

    Uses ``AutoVideoProcessor``, which requires transformers v5. v5 refactored
    VideoMAE attention to standard Linear layers and ships no mapping for the
    old ``q_bias``/``v_bias`` key names, so those weights load as zeros with no
    error; ``_restore_qkv_bias`` copies them back from the checkpoint. On v4,
    where the attention module still owns ``q_bias``, it is a no-op.
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
        from transformers import AutoModel, AutoVideoProcessor

        self.model_name = model_name
        self.revision = revision

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
        self.model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model.eval()
        self._restore_qkv_bias()

        self.model.to(self.device)

        # VideoMAE bakes the clip length into its temporal position embeddings
        # and transformers does not interpolate it, so the config is the source
        # of truth rather than a per-ModelMeta constant.
        self.num_frames = (
            num_frames if num_frames is not None else self.model.config.num_frames
        )
        if self.num_frames != self.model.config.num_frames:
            logger.warning(
                f"{model_name} was trained with {self.model.config.num_frames} frames "
                f"per clip, but num_frames={self.num_frames} was requested."
            )

    def _restore_qkv_bias(self) -> None:
        """Copy back q_bias/v_bias, which transformers v5 drops silently.

        v4 kept these as separate parameters with query/key/value at
        ``bias=False``, and that is how every published checkpoint is saved. v5
        uses standard Linears with their own biases and no key mapping, so they
        arrive zeroed. Skipped on v4, and on any version that loads them.
        """
        first = self.model.encoder.layer[0].attention.attention
        if hasattr(first, "q_bias") or first.query.bias is None:
            return
        if not bool((first.query.bias == 0).all()):
            return

        state_dict = _load_checkpoint_tensors(self.model_name, self.revision)
        restored = 0
        with torch.no_grad():
            for i, layer in enumerate(self.model.encoder.layer):
                attn = layer.attention.attention
                q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
                v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
                if q_key not in state_dict:
                    q_key = q_key.removeprefix("videomae.")
                    v_key = v_key.removeprefix("videomae.")
                if q_key not in state_dict or v_key not in state_dict:
                    continue
                attn.query.bias.copy_(state_dict[q_key])
                attn.value.bias.copy_(state_dict[v_key])
                attn.key.bias.zero_()  # k_bias was always an implicit zero
                restored += 1

        expected = len(self.model.encoder.layer)
        if restored != expected:
            raise RuntimeError(
                f"{self.model_name}: q_bias/v_bias were dropped on load but only "
                f"{restored}/{expected} layers could be restored. Refusing to "
                "return silently degraded embeddings."
            )
        logger.info("Restored q_bias/v_bias for %d VideoMAE layers", restored)

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
            # Explicit list-of-videos-of-frames in HWC. A list of 4D tensors trips
            # `make_batched` on the v4 slow processor, which treats the whole batch
            # as a single video and then fails inside PIL.
            processed = self.processor(padded, return_tensors="pt").to(self.device)

            outputs = self.model(**processed)
            pooled = outputs.last_hidden_state.mean(dim=1)
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
    extra_requirements_groups=["transformers-v5"],
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
    extra_requirements_groups=["transformers-v5"],
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
    extra_requirements_groups=["transformers-v5"],
)
