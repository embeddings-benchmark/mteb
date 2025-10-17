from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS13VisualSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS13VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts13",
            "revision": "561ee9ca47ff3e4a657283c59416deca8dc169f2",
        },
        description="SemEval STS 2013 dataset." + "then rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(eng)",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-12-31"),
        domains=["Web", "News", "Non-fiction", "Written"],
        task_subtypes=[],
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
