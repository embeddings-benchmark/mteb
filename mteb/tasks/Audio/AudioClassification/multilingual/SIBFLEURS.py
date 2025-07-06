from __future__ import annotations

from mteb.abstasks.Audio.AbsTaskAudioClassification import AbsTaskAudioClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

EVAL_LANGS_MAP = {
    "eng_Latn": ["eng-Latn"],  # English
    "tur_Latn": ["tur-Latn"],  # Turkish
    "fra_Latn": ["fra-Latn"],  # French
    "spa_Latn": ["spa-Latn"],  # Spanish
    "deu_Latn": ["deu-Latn"],  # German
    "arb_Arab": ["arb-Arab"],  # Arabic
    "hin_Deva": ["hin-Deva"],  # Hindi
    "rus_Cyrl": ["rus-Cyrl"],  # Russian
    "zho_Hans": ["zho-Hans"],  # Chinese (Simplified)
    "jpn_Jpan": ["jpn-Jpan"],  # Japanese
    "kor_Hang": ["kor-Hang"],  # Korean
    "ita_Latn": ["ita-Latn"],  # Italian
    "por_Latn": ["por-Latn"],  # Portuguese
    "nld_Latn": ["nld-Latn"],  # Dutch
    "pol_Latn": ["pol-Latn"],  # Polish
}


class SIBFLEURSMultilingualClassification(MultilingualTask, AbsTaskAudioClassification):
    metadata = TaskMetadata(
        name="SIBFLEURS",
        description="Topic Classification for multilingual audio dataset. This dataset is a stratified and downsampledsubset of the SIBFLEURS dataset, which is a collection of 1000+ hours of audio data in 100+ languages.",
        reference="https://huggingface.co/datasets/WueNLP/sib-fleurs",
        dataset={
            "path": "mteb/sib-fleurs-multilingual-mini",
            "revision": "6cc8ecb0b2892883f35a467e925211d6135d05e8",
        },
        type="AudioMultilabelClassification",
        category="a2t",
        eval_splits=["test"],
        eval_langs=EVAL_LANGS_MAP,
        main_score="accuracy",
        date=(
            "2024-12-09",
            "2024-12-13",
        ),
        domains=[
            "Encyclopaedic"
        ],  # original FLORES-101 dataset is read-out wikipedia corpus
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["audio"],
        sample_creation="found",
        bibtex_citation=r"""
@misc{schmidt2025fleursslumassivelymultilingualbenchmark,
  archiveprefix = {arXiv},
  author = {Fabian David Schmidt and Ivan Vulić and Goran Glavaš and David Ifeoluwa Adelani},
  eprint = {2501.06117},
  primaryclass = {cs.CL},
  title = {Fleurs-SLU: A Massively Multilingual Benchmark for Spoken Language Understanding},
  url = {https://arxiv.org/abs/2501.06117},
  year = {2025},
}
""",
        descriptive_stats={
            "n_samples": {"test": 177},
        },
    )

    audio_column_name: str = "audio"
    label_column_name: str = "category"
    samples_per_label: int = 10
