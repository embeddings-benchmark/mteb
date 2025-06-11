from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class Ocnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Ocnli",
        description="Original Chinese Natural Language Inference dataset",
        reference="https://arxiv.org/abs/2010.05444",
        dataset={
            "path": "C-MTEB/OCNLI",
            "revision": "66e76a618a34d6d565d5538088562851e6daa7ec",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="max_accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{hu2020ocnli,
  archiveprefix = {arXiv},
  author = {Hai Hu and Kyle Richardson and Liang Xu and Lu Li and Sandra Kuebler and Lawrence S. Moss},
  eprint = {2010.05444},
  primaryclass = {cs.CL},
  title = {OCNLI: Original Chinese Natural Language Inference},
  year = {2020},
}
""",
        prompt="Retrieve semantically similar text.",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")


class Cmnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Cmnli",
        description="Chinese Multi-Genre NLI",
        reference="https://huggingface.co/datasets/clue/viewer/cmnli",
        dataset={
            "path": "C-MTEB/CMNLI",
            "revision": "41bc36f332156f7adc9e38f53777c959b2ae9766",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="max_accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{xu-etal-2020-clue,
  address = {Barcelona, Spain (Online)},
  author = {Xu, Liang  and
Hu, Hai  and
Zhang, Xuanwei  and
Li, Lu  and
Cao, Chenjie  and
Li, Yudong  and
Xu, Yechen  and
Sun, Kai  and
Yu, Dian  and
Yu, Cong  and
Tian, Yin  and
Dong, Qianqian  and
Liu, Weitang  and
Shi, Bo  and
Cui, Yiming  and
Li, Junyi  and
Zeng, Jun  and
Wang, Rongzhao  and
Xie, Weijian  and
Li, Yanting  and
Patterson, Yina  and
Tian, Zuoyu  and
Zhang, Yiwen  and
Zhou, He  and
Liu, Shaoweihua  and
Zhao, Zhe  and
Zhao, Qipeng  and
Yue, Cong  and
Zhang, Xinrui  and
Yang, Zhengliang  and
Richardson, Kyle  and
Lan, Zhenzhong},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  doi = {10.18653/v1/2020.coling-main.419},
  month = dec,
  pages = {4762--4772},
  publisher = {International Committee on Computational Linguistics},
  title = {{CLUE}: A {C}hinese Language Understanding Evaluation Benchmark},
  url = {https://aclanthology.org/2020.coling-main.419},
  year = {2020},
}
""",
        prompt="Retrieve semantically similar text.",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
