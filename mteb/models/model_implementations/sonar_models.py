from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput


class SONARWrapper(AbsEncoder):
    """MTEB wrapper for Meta's SONAR text + speech embedding model."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        text_encoder: str = "text_sonar_basic_encoder",
        speech_encoder: str = "sonar_speech_encoder_eng",
        **kwargs: Any,
    ) -> None:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.text_model = TextToEmbeddingModelPipeline(
            encoder=text_encoder,
            tokenizer=text_encoder,
            device=torch.device(self.device),
        )
        self.speech_model = SpeechToEmbeddingModelPipeline(
            encoder=speech_encoder,
            device=torch.device(self.device),
        )

    @staticmethod
    def _to_sonar_lang(
        task_metadata: TaskMetadata,
        hf_subset: str | None,
        prompt_type: PromptType | None,
    ) -> str:
        """Resolve the SONAR-format language code ('eng_Latn') for this batch.

        Uses the task's own eval_langs rather than parsing hf_subset, since
        subset names have no guaranteed format. For bitext subsets this list
        has two entries (source, target); queries use the source side,
        documents use the target side. eval_langs uses BCP47 ('eng-Latn'),
        SONAR wants underscore-joined FLORES-200-style codes ('eng_Latn').

        Tasks with `parallel_subsets=True` (e.g. FLORES) instead pass a
        single already-resolved language code as hf_subset -- one per
        dataset column, not a key into eval_langs -- so it's used as-is.
        """
        langscripts = task_metadata.hf_subsets_to_langscripts
        if hf_subset in langscripts:
            langs = langscripts[hf_subset]
            lang = langs[-1] if prompt_type == PromptType.document else langs[0]
        else:
            lang = hf_subset or "eng_Latn"
        return lang.replace("-", "_")

    def _encode_audio_batch(self, audio_items: list) -> torch.Tensor:
        import soundfile as sf

        embeddings = []
        for item in audio_items:
            array = np.asarray(item["array"])
            sr = item["sampling_rate"]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, array, sr)
                emb = self.speech_model.predict([tmp.name])
            embeddings.append(emb[0])
        return torch.stack(embeddings)

    @torch.inference_mode()
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
        has_text = "text" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features
        source_lang = self._to_sonar_lang(task_metadata, hf_subset, prompt_type)

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            embs: list[torch.Tensor] = []
            if has_text and batch.get("text"):
                text_emb = self.text_model.predict(
                    batch["text"], source_lang=source_lang
                )
                embs.append(text_emb)
            if has_audio and batch.get("audio"):
                audio_emb = self._encode_audio_batch(batch["audio"])
                embs.append(audio_emb)

            if not embs:
                raise ValueError(
                    f"No supported modality found in batch: {list(batch.keys())}"
                )
            batch_emb = embs[0] if len(embs) == 1 else sum(embs[1:], embs[0])
            all_embeddings.append(batch_emb.cpu())

        return torch.cat(all_embeddings, dim=0).float().numpy()


SONAR_CITATION = """@misc{Duquenne:2023:sonar_arxiv,
  author = {Paul-Ambroise Duquenne and Holger Schwenk and Benoit Sagot},
  title = {{SONAR:} Sentence-Level Multimodal and Language-Agnostic Representations},
  publisher = {arXiv},
  year = {2023},
  url = {https://arxiv.org/abs/2308.11466},
}"""

sonar_langs = [
    "ace-Arab",
    "ace-Latn",
    "acm-Arab",
    "acq-Arab",
    "aeb-Arab",
    "afr-Latn",
    "ajp-Arab",
    "aka-Latn",
    "als-Latn",
    "amh-Ethi",
    "apc-Arab",
    "arb-Arab",
    "arb-Latn",
    "ars-Arab",
    "ary-Arab",
    "arz-Arab",
    "asm-Beng",
    "ast-Latn",
    "awa-Deva",
    "ayr-Latn",
    "azb-Arab",
    "azj-Latn",
    "bak-Cyrl",
    "bam-Latn",
    "ban-Latn",
    "bel-Cyrl",
    "bem-Latn",
    "ben-Beng",
    "bho-Deva",
    "bjn-Arab",
    "bjn-Latn",
    "bod-Tibt",
    "bos-Latn",
    "bug-Latn",
    "bul-Cyrl",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cjk-Latn",
    "ckb-Arab",
    "crh-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "dik-Latn",
    "dyu-Latn",
    "dzo-Tibt",
    "ell-Grek",
    "eng-Latn",
    "epo-Latn",
    "est-Latn",
    "eus-Latn",
    "ewe-Latn",
    "fao-Latn",
    "fij-Latn",
    "fin-Latn",
    "fon-Latn",
    "fra-Latn",
    "fur-Latn",
    "fuv-Latn",
    "gaz-Latn",
    "gla-Latn",
    "gle-Latn",
    "glg-Latn",
    "grn-Latn",
    "guj-Gujr",
    "hat-Latn",
    "hau-Latn",
    "heb-Hebr",
    "hin-Deva",
    "hne-Deva",
    "hrv-Latn",
    "hun-Latn",
    "hye-Armn",
    "ibo-Latn",
    "ilo-Latn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jav-Latn",
    "jpn-Jpan",
    "kab-Latn",
    "kac-Latn",
    "kam-Latn",
    "kan-Knda",
    "kas-Arab",
    "kas-Deva",
    "kat-Geor",
    "kaz-Cyrl",
    "kbp-Latn",
    "kea-Latn",
    "khk-Cyrl",
    "khm-Khmr",
    "kik-Latn",
    "kin-Latn",
    "kir-Cyrl",
    "kmb-Latn",
    "kmr-Latn",
    "knc-Arab",
    "knc-Latn",
    "kon-Latn",
    "kor-Hang",
    "lao-Laoo",
    "lij-Latn",
    "lim-Latn",
    "lin-Latn",
    "lit-Latn",
    "lmo-Latn",
    "ltg-Latn",
    "ltz-Latn",
    "lua-Latn",
    "lug-Latn",
    "luo-Latn",
    "lus-Latn",
    "lvs-Latn",
    "mag-Deva",
    "mai-Deva",
    "mal-Mlym",
    "mar-Deva",
    "min-Arab",
    "min-Latn",
    "mkd-Cyrl",
    "mlt-Latn",
    "mni-Beng",
    "mos-Latn",
    "mri-Latn",
    "mya-Mymr",
    "nld-Latn",
    "nno-Latn",
    "nob-Latn",
    "npi-Deva",
    "nso-Latn",
    "nus-Latn",
    "nya-Latn",
    "oci-Latn",
    "ory-Orya",
    "pag-Latn",
    "pan-Guru",
    "pap-Latn",
    "pbt-Arab",
    "pes-Arab",
    "plt-Latn",
    "pol-Latn",
    "por-Latn",
    "prs-Arab",
    "quy-Latn",
    "ron-Latn",
    "run-Latn",
    "rus-Cyrl",
    "sag-Latn",
    "san-Deva",
    "sat-Olck",
    "scn-Latn",
    "shn-Mymr",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "smo-Latn",
    "sna-Latn",
    "snd-Arab",
    "som-Latn",
    "sot-Latn",
    "spa-Latn",
    "srd-Latn",
    "srp-Cyrl",
    "ssw-Latn",
    "sun-Latn",
    "swe-Latn",
    "swh-Latn",
    "szl-Latn",
    "tam-Taml",
    "taq-Latn",
    "taq-Tfng",
    "tat-Cyrl",
    "tel-Telu",
    "tgk-Cyrl",
    "tgl-Latn",
    "tha-Thai",
    "tir-Ethi",
    "tpi-Latn",
    "tsn-Latn",
    "tso-Latn",
    "tuk-Latn",
    "tum-Latn",
    "tur-Latn",
    "twi-Latn",
    "tzm-Tfng",
    "uig-Arab",
    "ukr-Cyrl",
    "umb-Latn",
    "urd-Arab",
    "uzn-Latn",
    "vec-Latn",
    "vie-Latn",
    "war-Latn",
    "wol-Latn",
    "xho-Latn",
    "ydd-Hebr",
    "yor-Latn",
    "yue-Hant",
    "zho-Hans",
    "zho-Hant",
    "zsm-Latn",
    "zul-Latn",
]

sonar = ModelMeta(
    loader=SONARWrapper,
    name="facebook/SONAR",
    model_type=["dense"],
    languages=sonar_langs,
    open_weights=True,
    use_instructions=False,  # it does take a language code as input
    revision="a551c586dcf4a49c8fd847de369412d556a7f2f2",
    release_date="2021-05-21",
    n_parameters=None,
    n_embedding_parameters=None,  # it is really multiple models so not sure how to calculate this
    max_tokens=512,  # https://github.com/facebookresearch/SONAR/blob/549d287466443bd8720f938047882630c1c5c3f7/sonar/models/sonar_text/builder.py#L139
    embed_dim=1024,
    license="cc-by-nc-4.0",
    memory_usage_mb=None,
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["PyTorch"],
    modalities=["text", "audio"],
    reference="https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/",
    training_datasets=set(
        # "FloresBitextMining", # I believe it only used for evaluation
        # "IndicGenBenchFloresBitextMining", # extension of Flores so I would say not trained on
    ),
    public_training_code="https://github.com/facebookresearch/SONAR",
    public_training_data=None,  # couldn't find this
    citation=SONAR_CITATION,
    extra_requirements_groups=["sonar"],
)
