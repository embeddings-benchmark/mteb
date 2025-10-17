from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class STS16VisualSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STS16VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts16",
            "revision": "fc354f19598af93f32c0af1b94046ffdeaacde15",
        },
        description="SemEval STS 2016 dataset" + "rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS(eng)",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2015-10-01", "2015-12-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
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
