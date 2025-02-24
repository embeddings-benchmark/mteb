from __future__ import annotations

from mteb.abstasks.Image.AbsTaskVisualSTS import AbsTaskVisualSTS
from mteb.abstasks.TaskMetadata import TaskMetadata


class STS12VisualSTS(AbsTaskVisualSTS):
    metadata = TaskMetadata(
        name="STS12VisualSTS",
        dataset={
            "path": "Pixel-Linguist/rendered-sts12",
            "revision": "820c25edfba736f3789201b2476208cc62c2ccb9",
        },
        description="SemEval-2012 Task 6." + "then rendered into images.",
        reference="https://arxiv.org/abs/2402.08183/",
        type="VisualSTS",
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
        bibtex_citation="""@article{xiao2024pixel,
  title={Pixel Sentence Representation Learning},
  author={Xiao, Chenghao and Huang, Zhuoxu and Chen, Danlu and Hudson, G Thomas and Li, Yizhi and Duan, Haoran and Lin, Chenghua and Fu, Jie and Han, Jungong and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.08183},
  year={2024}
}""",
    )
