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


class _FixedFrameVideoWrapper(AbsEncoder):
    """Shared base for video-only encoders with a fixed input clip length.

    VideoMAE and TimeSformer both bake the clip length into their temporal
    position embeddings, so the number of frames handed to the model has to
    equal ``config.num_frames`` for that checkpoint. Rather than restate it in
    every ``ModelMeta``, we read it off the loaded config and pass it to
    ``FramesCollator`` in fixed-sample mode.

    Note on the processor: both families are loaded with ``AutoImageProcessor``
    rather than ``AutoVideoProcessor``. ``VideoMAEVideoProcessor`` and
    ``TimesformerVideoProcessor`` only exist from transformers v5 onwards,
    while ``mteb`` supports ``transformers>=4.40``. Every published checkpoint
    of both families ships a ``preprocessor_config.json`` declaring
    ``VideoMAEImageProcessor``, which resolves on both v4 and v5 and carries
    the checkpoint's own resize/crop/normalisation values.
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

        self._post_load()

        self.model.to(self.device)

        # The checkpoint config is the source of truth for clip length.
        self.num_frames = (
            num_frames if num_frames is not None else self.model.config.num_frames
        )
        if self.num_frames != self.model.config.num_frames:
            raise ValueError(
                f"{model_name} was trained with {self.model.config.num_frames} frames "
                f"per clip, but num_frames={self.num_frames} was requested. The "
                "temporal position embeddings are not interpolated, so this would "
                "either error or silently degrade the embedding."
            )

    def _post_load(self) -> None:
        """Hook for per-family weight fixups. Runs before the model is moved to device."""

    def _pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
            # FramesCollator repeats frames for clips shorter than num_frames,
            # so every clip already has the right length. Kept as a guard in
            # case a task hands back a short decode.
            padded = [
                torch.cat(
                    [
                        v,
                        v[-1:].expand(self.num_frames - v.shape[0], *v.shape[1:]),
                    ],
                    dim=0,
                )
                if v.shape[0] < self.num_frames
                else v[: self.num_frames]
                for v in videos
            ]
            # Positional call: `VideoMAEImageProcessor.__call__` takes the batch
            # as its first positional argument on both transformers v4 and v5.
            processed = self.processor(padded, return_tensors="pt").to(self.device)

            outputs = self.model(**processed)
            pooled = self._pool(outputs.last_hidden_state)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0).numpy()


class VideoMAEWrapper(_FixedFrameVideoWrapper):
    """VideoMAE encoder.

    Two fixups happen in `_post_load`, both of which change the output.

    1. **Attention biases.** transformers 4.x implemented VideoMAE attention
       BEiT-style: `query`/`key`/`value` Linears with `bias=False`, plus separate
       `q_bias` and `v_bias` parameters and an implicit zero `k_bias`. Every
       published VideoMAE checkpoint stores the biases under those names.
       transformers 5.x refactored to standard Linears with
       `bias=config.qkv_bias` and shipped no conversion mapping, so on v5 the
       stored biases are dropped and the new ones are zero-initialised. The
       model loads without error and returns a perturbed embedding: the stored
       biases are far from zero (max |value| 3.04 across the encoder) and
       restoring them moves the pooled embedding by 10% in relative L2
       (cosine 0.9950). We copy them back. On v4 the attention
       module still has a `q_bias` attribute and this is skipped, and if a
       future transformers adds the conversion the loaded bias will be
       non-zero and this is skipped too.

    2. **Final normalisation.** `VideoMAEModel.layernorm` exists only when
       `config.use_mean_pooling` is False. The self-supervised checkpoints
       (`videomae-base`, `videomae-large`) set it False, so their final
       LayerNorm lives inside the base model and is applied for us. The
       finetuned checkpoints set it True, which makes `VideoMAEModel.layernorm`
       None and moves the LayerNorm to `fc_norm` on the classification head,
       where `AutoModel` discards it. Without this fixup the two groups would
       produce embeddings from different pipelines. We load `fc_norm` from the
       checkpoint and apply it after mean pooling, exactly as
       `VideoMAEForVideoClassification` does.

    VideoMAE has no CLS token, so pooling is a mean over the token axis.
    """

    def _post_load(self) -> None:
        state_dict: dict[str, torch.Tensor] | None = None

        def tensors() -> dict[str, torch.Tensor]:
            nonlocal state_dict
            if state_dict is None:
                state_dict = _load_checkpoint_tensors(self.model_name, self.revision)
            return state_dict

        self._restore_qkv_bias(tensors)
        self._load_fc_norm(tensors)

    def _restore_qkv_bias(self, tensors: Any) -> None:
        first = self.model.encoder.layer[0].attention.attention
        if hasattr(first, "q_bias"):
            return  # transformers v4 layout: biases loaded natively
        if first.query.bias is None:
            return  # config.qkv_bias is False, nothing to restore
        if not bool((first.query.bias == 0).all()):
            return  # already populated, e.g. a fixed future transformers

        sd = tensors()
        restored = 0
        with torch.no_grad():
            for i, layer in enumerate(self.model.encoder.layer):
                attn = layer.attention.attention
                q_key = f"videomae.encoder.layer.{i}.attention.attention.q_bias"
                v_key = f"videomae.encoder.layer.{i}.attention.attention.v_bias"
                if q_key not in sd:
                    q_key = q_key.removeprefix("videomae.")
                    v_key = v_key.removeprefix("videomae.")
                if q_key not in sd or v_key not in sd:
                    continue
                attn.query.bias.copy_(sd[q_key])
                attn.value.bias.copy_(sd[v_key])
                attn.key.bias.zero_()  # k_bias was always an implicit zero
                restored += 1

        expected = len(self.model.encoder.layer)
        if restored != expected:
            raise RuntimeError(
                f"{self.model_name}: this transformers version dropped VideoMAE's "
                f"q_bias/v_bias during load, but only {restored}/{expected} layers "
                "could be restored from the checkpoint. Refusing to return silently "
                "degraded embeddings. Pin transformers<5 or report this upstream."
            )
        logger.info(
            "Restored q_bias/v_bias for %d VideoMAE layers (transformers v5 "
            "dropped them during checkpoint load)",
            restored,
        )

    def _load_fc_norm(self, tensors: Any) -> None:
        self.fc_norm: torch.nn.LayerNorm | None = None
        if not getattr(self.model.config, "use_mean_pooling", False):
            return  # base model keeps its own layernorm; nothing to add

        sd = tensors()
        if "fc_norm.weight" not in sd:
            logger.warning(
                "%s sets use_mean_pooling=True but has no fc_norm in the "
                "checkpoint; embeddings will be un-normalised.",
                self.model_name,
            )
            return

        fc_norm = torch.nn.LayerNorm(self.model.config.hidden_size)
        fc_norm.load_state_dict(
            {"weight": sd["fc_norm.weight"], "bias": sd["fc_norm.bias"]}
        )
        fc_norm.eval()
        self.fc_norm = fc_norm.to(self.device)

    def _pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        pooled = last_hidden_state.mean(dim=1)
        if self.fc_norm is not None:
            pooled = self.fc_norm(pooled)
        return pooled


class TimesformerWrapper(_FixedFrameVideoWrapper):
    """TimeSformer encoder.

    Unlike VideoMAE, TimeSformer prepends a CLS token and
    ``TimesformerForVideoClassification`` reads ``last_hidden_state[:, 0]``.
    We pool the same way so the embedding matches what the model was trained
    to make linearly separable. These checkpoints load cleanly on both
    transformers v4 and v5; only the classification head is discarded.
    """

    def _pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:  # noqa: PLR6301
        return last_hidden_state[:, 0]


_VIDEOMAE_CITATION = """
@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}"""

_TIMESFORMER_CITATION = """
@inproceedings{bertasius2021space,
  title={Is Space-Time Attention All You Need for Video Understanding?},
  author={Bertasius, Gedas and Wang, Heng and Torresani, Lorenzo},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2021},
}"""

# VideoMAE self-supervised pre-training uses Kinetics-400 *videos* without
# labels. Declared so the leaderboard flags the overlap even though no label
# was seen.
_K400_UNLABELED = {
    "Kinetics400V",
    "Kinetics400VA",
    "Kinetics400ZeroShot",
    "Kinetics400VAZeroShot",
}

_K400_SUPERVISED = _K400_UNLABELED
_SSV2_SUPERVISED = {"SomethingSomethingV2Classification"}


# ---------------------------------------------------------------------------
# VideoMAE
# ---------------------------------------------------------------------------

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
    training_datasets=_K400_UNLABELED,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
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
    training_datasets=_K400_UNLABELED,
    adapted_from=None,
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
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
    training_datasets=_K400_SUPERVISED,
    adapted_from="MCG-NJU/videomae-base",
    superseded_by=None,
    modalities=["video"],
    model_type=["dense"],
    citation=_VIDEOMAE_CITATION,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)


# ---------------------------------------------------------------------------
# TimeSformer
#
# Every published TimeSformer checkpoint is supervised-finetuned on
# Kinetics-400, Kinetics-600, or Something-Something V2. There is no
# pretrained-only variant, so the corresponding mteb tasks are declared in
# `training_datasets` for all of them.
# ---------------------------------------------------------------------------

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
