from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS12VisualSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS12VisualSTS",
        dataset={
            "path": "mteb/rendered-sts12",
            "revision": "16fabd3185c013b2e5e73f8b3343068e3d438689",
        },
        description="SemEval-2012 Task 6." + "then rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(eng)",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2005-01-01", "2012-12-31"),
        domains=["Encyclopaedic", "News", "Written"],
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
