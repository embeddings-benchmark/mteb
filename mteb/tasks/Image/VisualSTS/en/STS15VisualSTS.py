from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class STS15VisualSTS(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        name="STS15VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts15",
            "revision": "1f8d08d9b9daac7118dfdefeb94b0aac4baf2e5f",
        },
        description="SemEval STS 2015 dataset" + "rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS",
        category="i2i",
        modalities=["image"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2008-01-01", "2014-07-28"),
        domains=["Blog", "News", "Web", "Written", "Spoken"],
        task_subtypes=[],
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
    )
