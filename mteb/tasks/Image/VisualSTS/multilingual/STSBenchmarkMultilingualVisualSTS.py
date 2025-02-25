from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
    "nl": ["nld-Latn"],
    "pl": ["pol-Latn"],
    "pt": ["por-Latn"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
}

_SPLITS = ["dev", "test"]


class STSBenchmarkMultilingualVisualSTS(AbsTaskVisualSTS, MultilingualTask):
    metadata = TaskMetadata(
        name="STSBenchmarkMultilingualVisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-stsb",
            "revision": "9f1ab21f17f497974996ab74b3ff911165a7dbf9",
        },
        description=(
            "Semantic Textual Similarity Benchmark (STSbenchmark) dataset, "
            + "translated into target languages using DeepL API,"
            + "then rendered into images."
            + "built upon multi-sts created by Philip May"
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
            "n_samples": {"dev": 15000, "test": 13790},
            "avg_character_length": {"dev": 1.0, "test": 1.0},
        },
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
