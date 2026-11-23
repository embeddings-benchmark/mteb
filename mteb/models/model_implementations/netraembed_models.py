from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import OutputDType

from .colpali_models import COLPALI_TRAINING_DATA, ColPaliEngineWrapper

if TYPE_CHECKING:
    import torch

# Both checkpoints were saved with transformers 4.57, which names the language model's
# weights `language_model.model.*`. transformers 5 only renames these for its own Gemma3
# classes: without this mapping `ColGemma3` leaves the language model randomly initialized
# and `BiGemma3` fails to load (colpali-engine 0.3.18, transformers 5.20.2). The
# `netraembed` extra pins transformers to the range this was tested on.
_KEY_MAPPING = {r"^language_model\.model": "language_model"}

# Queries keep the shared wrapper's ten `<eos>` augmentation tokens. colpali-engine's
# training collator adds them to every query, though the model cards' examples do not.


class ColNetraEmbedWrapper(ColPaliEngineWrapper):
    """Wrapper for ColNetraEmbed, a late-interaction model built on Gemma 3."""

    def __init__(
        self,
        model_name: str = "Cognitive-Lab/ColNetraEmbed",
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        from colpali_engine.models import ColGemma3, ColGemmaProcessor3

        super().__init__(
            model_name=model_name,
            model_class=ColGemma3,
            processor_class=ColGemmaProcessor3,
            revision=revision,
            device=device,
            # the repository holds the full weights, so `revision` pins the model itself
            base_revision=revision,
            key_mapping=_KEY_MAPPING,
            **kwargs,
        )


class NetraEmbedWrapper(ColPaliEngineWrapper):
    """Wrapper for NetraEmbed, a single-vector model built on Gemma 3."""

    def __init__(
        self,
        model_name: str = "Cognitive-Lab/NetraEmbed",
        revision: str | None = None,
        device: str | None = None,
        embed_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        from colpali_engine.models import BiGemma3, BiGemmaProcessor3

        super().__init__(
            model_name=model_name,
            model_class=BiGemma3,
            processor_class=BiGemmaProcessor3,
            revision=revision,
            device=device,
            # the repository holds the full weights, so `revision` pins the model itself
            base_revision=revision,
            key_mapping=_KEY_MAPPING,
            **kwargs,
        )
        self.embed_dim = embed_dim

    def encode_input(self, inputs: dict[str, Any]) -> torch.Tensor:
        if self.embed_dim is None:
            return self.mdl(**inputs)
        # Matryoshka: the model truncates and re-normalizes the embedding itself
        return self.mdl(**inputs, embedding_dim=self.embed_dim)


NETRAEMBED_CITATION = r"""@misc{kolavi2025m3druniversalmultilingualmultimodal,
  archiveprefix = {arXiv},
  author = {Adithya S Kolavi and Vyoman Jain},
  eprint = {2512.03514},
  primaryclass = {cs.IR},
  title = {M3DR: Towards Universal Multilingual Multimodal Document Retrieval},
  url = {https://arxiv.org/abs/2512.03514},
  year = {2025},
}"""

NETRAEMBED_LANGUAGES = [
    "eng-Latn",
    "spa-Latn",
    "fra-Latn",
    "deu-Latn",
    "ita-Latn",
    "hin-Deva",
    "mar-Deva",
    "san-Deva",
    "kan-Knda",
    "tel-Telu",
    "tam-Taml",
    "mal-Mlym",
    "zho-Hans",
    "jpn-Jpan",
    "kor-Hang",
    "ara-Arab",
    "ben-Beng",
    "guj-Gujr",
    "ory-Orya",
    "pan-Guru",
    "rus-Cyrl",
    "tha-Thai",
]

# Both models were trained on the ColPali train set and then on about 250k synthetic
# Nayana IR pairs in 22 languages (Table 17 of the paper). The Nayana-IR test sets are
# held out from that data but come from the same pipeline, so they are listed here.
NETRAEMBED_TRAINING_DATA = COLPALI_TRAINING_DATA | {
    "NayanaIRMonoBenchTamilRetrieval",
    "NayanaIRMonoBenchKannadaRetrieval",
    "NayanaIRMonoBenchGujaratiRetrieval",
    "NayanaIRMonoBenchPunjabiRetrieval",
    "NayanaIRMonoBenchMalayalamRetrieval",
    "NayanaIRMonoBenchMarathiRetrieval",
    "NayanaIRMonoBenchOdiaRetrieval",
    "NayanaIRMonoBenchSanskritRetrieval",
}

colnetraembed = ModelMeta(
    loader=ColNetraEmbedWrapper,
    loader_kwargs=dict(
        torch_dtype=OutputDType.BF16,
    ),
    name="Cognitive-Lab/ColNetraEmbed",
    model_type=["late-interaction"],
    languages=NETRAEMBED_LANGUAGES,
    revision="a27fe0bdcf65a54116393853621f7f80e3cd9ca3",
    release_date="2025-12-01",
    modalities=["image", "text"],
    n_parameters=4_300_407_280,
    n_embedding_parameters=671_252_480,  # vocab 262208 x hidden 2560
    memory_usage_mb=8202,
    max_tokens=131072,
    embed_dim=128,
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/Cognitive-Lab/ColNetraEmbed",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=NETRAEMBED_TRAINING_DATA,
    adapted_from="google/gemma-3-4b-it",
    citation=NETRAEMBED_CITATION,
    extra_requirements_groups=["netraembed"],
)

netraembed = ModelMeta(
    loader=NetraEmbedWrapper,
    loader_kwargs=dict(
        torch_dtype=OutputDType.BF16,
    ),
    name="Cognitive-Lab/NetraEmbed",
    model_type=["dense"],
    languages=NETRAEMBED_LANGUAGES,
    revision="4176488e9bae6c7663d8b2c8936bde2473520da2",
    release_date="2025-12-01",
    modalities=["image", "text"],
    n_parameters=4_300_079_472,
    n_embedding_parameters=671_252_480,  # vocab 262208 x hidden 2560
    memory_usage_mb=8202,
    max_tokens=131072,
    # Matryoshka: results in the paper are reported at the full 2560 width.
    embed_dim=[768, 1536, 2560],
    license="gemma",
    open_weights=True,
    public_training_code="https://github.com/adithya-s-k/colpali",
    public_training_data=None,
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/Cognitive-Lab/NetraEmbed",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=NETRAEMBED_TRAINING_DATA,
    adapted_from="google/gemma-3-4b-it",
    citation=NETRAEMBED_CITATION,
    extra_requirements_groups=["netraembed"],
)
