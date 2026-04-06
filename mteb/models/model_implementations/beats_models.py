"""BEATs: Audio Pre-Training with Acoustic Tokenizers.

Reference: https://github.com/microsoft/unilm/tree/master/beats

Setup:
    BEATs is not available as a pip package. The source files (BEATs.py,
    backbone.py, modules.py) are automatically downloaded from the unilm
    repository to ``~/.cache/torch/hub/beats_src/`` on first use.

    Pretrained checkpoint weights must be downloaded manually from OneDrive
    and passed as ``model_name`` when constructing the wrapper:

        - BEATs_iter1: https://1drv.ms/u/s!AqeByhGUtINrgcpmY7IHhgc9q0pT7Q
        - BEATs_iter2: https://1drv.ms/u/s!AqeByhGUtINrgcpwwEGgUyiI-jQyQw
        - BEATs_iter3: https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA

    Example::

        from mteb.models.model_implementations.beats_models import beats_iter2

        beats_iter2.name = "/path/to/BEATs_iter2.pt"
        results = mteb.evaluate(beats_iter2, tasks)
"""

from __future__ import annotations

import importlib
import logging
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._create_dataloaders import AudioCollator
from mteb._requires_package import requires_audio_dependencies
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput

logger = logging.getLogger(__name__)

_BEATS_SRC_URL = "https://raw.githubusercontent.com/microsoft/unilm/732d834db70ee0fc3886b4bcbcfb4ce7fb829be2/beats"
_BEATS_SRC_FILES = ["BEATs.py", "backbone.py", "modules.py"]

_BEATS_CITATION = """
@article{Chen2022beats,
  title = {BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author = {Sanyuan Chen and Yu Wu and Chengyi Wang and Shujie Liu and Daniel Tompkins and Zhuo Chen and Furu Wei},
  eprint = {2212.09058},
  archiveprefix = {arXiv},
  year = {2022},
}
"""


def _load_beats_modules():
    """Download BEATs source files if needed and import the module."""
    try:
        from BEATs import BEATs, BEATsConfig

        return BEATs, BEATsConfig
    except ImportError:
        pass

    cache_dir = Path(torch.hub.get_dir()) / "beats_src"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for fname in _BEATS_SRC_FILES:
        dest = cache_dir / fname
        if not dest.exists():
            logger.info("Downloading %s to %s", fname, dest)
            url = f"{_BEATS_SRC_URL}/{fname}"
            urllib.request.urlretrieve(url, dest)  # noqa: S310

    if str(cache_dir) not in sys.path:
        sys.path.insert(0, str(cache_dir))

    for mod_name in ["BEATs", "backbone", "modules"]:
        sys.modules.pop(mod_name, None)

    beats_mod = importlib.import_module("BEATs")
    return beats_mod.BEATs, beats_mod.BEATsConfig


class BEATsWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        requires_audio_dependencies()
        beats_cls, beats_config_cls = _load_beats_modules()

        self.model_name = model_name
        self.device = device
        self.sampling_rate = 16_000

        checkpoint = torch.load(model_name, map_location=device, weights_only=False)
        cfg = beats_config_cls(checkpoint["cfg"])
        self.model = beats_cls(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(self.sampling_rate)
        all_embeddings = []

        for batch in tqdm(inputs, disable=not show_progress_bar):
            audio_tensors = []
            lengths = []
            for a in batch["audio"]:
                array = a["array"]
                if not isinstance(array, torch.Tensor):
                    array = torch.tensor(array, dtype=torch.float32)
                else:
                    array = array.float()
                audio_tensors.append(array)
                lengths.append(len(array))

            max_len = max(lengths)
            padded = torch.zeros(len(audio_tensors), max_len)
            padding_mask = torch.ones(len(audio_tensors), max_len, dtype=torch.bool)
            for i, (tensor, length) in enumerate(zip(audio_tensors, lengths)):
                padded[i, :length] = tensor
                padding_mask[i, :length] = False

            padded = padded.to(self.device)
            padding_mask = padding_mask.to(self.device)

            representation = self.model.extract_features(
                padded, padding_mask=padding_mask
            )[0]

            embeddings = representation.mean(dim=1)
            all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()

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
        if "audio" not in inputs.dataset.features:
            raise ValueError("BEATsWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


_common_meta = dict(
    languages=["eng-Latn"],
    open_weights=True,
    revision="732d834db70ee0fc3886b4bcbcfb4ce7fb829be2",
    release_date="2022-12-18",
    max_tokens=None,
    n_parameters=90_000_000,
    n_embedding_parameters=0,
    memory_usage_mb=350,
    embed_dim=768,
    license="mit",
    reference="https://arxiv.org/abs/2212.09058",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/microsoft/unilm/tree/master/beats",
    public_training_data="https://research.google.com/audioset/dataset/index.html",
    training_datasets={"AudioSetMini"},
    modalities=["audio"],
    citation=_BEATS_CITATION,
)

beats_iter1 = ModelMeta(
    loader=BEATsWrapper,
    name="microsoft/beats-iter1",
    **_common_meta,
)

beats_iter2 = ModelMeta(
    loader=BEATsWrapper,
    name="microsoft/beats-iter2",
    **_common_meta,
)

beats_iter3 = ModelMeta(
    loader=BEATsWrapper,
    name="microsoft/beats-iter3",
    **_common_meta,
)
