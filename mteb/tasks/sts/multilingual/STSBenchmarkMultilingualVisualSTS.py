from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

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


class STSBenchmarkMultilingualVisualSTS(AbsTaskSTS):
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
        bibtex_citation=r"""
@article{xiao2024pixel,
  author = {Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal = {arXiv preprint arXiv:2402.08183},
  title = {Pixel Sentence Representation Learning},
  year = {2024},
}
""",
    )
