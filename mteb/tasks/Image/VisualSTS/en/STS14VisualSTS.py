from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class STS14VisualSTS(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        name="STS14VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts14",
            "revision": "824e95e45471024a684b901e0645579ffd9ca288",
        },
        description="SemEval STS 2014 dataset. Currently only the English dataset."
        + "rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS",
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
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
