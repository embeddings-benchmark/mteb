from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS14VisualSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS14VisualSTS",
        dataset={
            "path": "mteb/rendered-sts14",
            "revision": "0a54520fac778d76f3009f7cb7290ad4c7ffc678",
        },
        description="SemEval STS 2014 dataset. Currently only the English dataset."
        + "rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(eng)",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-08-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
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
