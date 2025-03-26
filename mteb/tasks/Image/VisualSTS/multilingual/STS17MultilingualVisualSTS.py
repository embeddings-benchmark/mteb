from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "ko-ko": ["kor-Hang"],
    "ar-ar": ["ara-Arab"],
    "en-ar": ["eng-Latn", "ara-Arab"],
    "en-de": ["eng-Latn", "deu-Latn"],
    "en-en": ["eng-Latn"],
    "en-tr": ["eng-Latn", "tur-Latn"],
    "es-en": ["spa-Latn", "eng-Latn"],
    "es-es": ["spa-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "it-en": ["ita-Latn", "eng-Latn"],
    "nl-en": ["nld-Latn", "eng-Latn"],
}

_SPLITS = ["test"]


class STS17MultilingualVisualSTS(AbsTaskVisualSTS, MultilingualTask):
    metadata = TaskMetadata(
        name="STS17MultilingualVisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts17",
            "revision": "2e31b4b459551a51e1ab54fd7266b40f3fe510d4",
        },
        description=(
            "Semantic Textual Similarity 17 (STS-17) dataset, "
            + "rendered into images."
        ),
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(multi)",
        category="i2i",
        modalities=["image"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2012-01-01", "2017-12-31"),
        domains=["News", "Social", "Web", "Spoken", "Written"],
        task_subtypes=["Rendered semantic textual similarity"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="rendered",
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
        descriptive_stats={
            "n_samples": {"test": 10692},
            "avg_character_length": {"dev": 1.0, "test": 1.0},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
