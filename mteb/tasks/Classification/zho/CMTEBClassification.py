from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TNews",
        description="Short Text Classification for News",
        reference="https://www.cluebenchmarks.com/introduce.html",
        dataset={
            "path": "C-MTEB/TNews-classification",
            "revision": "317f262bf1e6126357bbe89e875451e4b0938fe4",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
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
Hu, Hai and
Zhang, Xuanwei and
Li, Lu and
Cao, Chenjie and
Li, Yudong and
Xu, Yechen and
Sun, Kai and
Yu, Dian and
Yu, Cong and
Tian, Yin and
Dong, Qianqian and
Liu, Weitang and
Shi, Bo and
Cui, Yiming and
Li, Junyi and
Zeng, Jun and
Wang, Rongzhao and
Xie, Weijian and
Li, Yanting and
Patterson, Yina and
Tian, Zuoyu and
Zhang, Yiwen and
Zhou, He and
Liu, Shaoweihua and
Zhao, Zhe and
Zhao, Qipeng and
Yue, Cong and
Zhang, Xinrui and
Yang, Zhengliang and
Richardson, Kyle and
Lan, Zhenzhong },
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
        prompt="Classify the fine-grained category of the given news title",
    )

    samples_per_label = 32


class IFlyTek(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IFlyTek",
        description="Long Text classification for the description of Apps",
        reference="https://www.cluebenchmarks.com/introduce.html",
        dataset={
            "path": "C-MTEB/IFlyTek-classification",
            "revision": "421605374b29664c5fc098418fe20ada9bd55f8a",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{xu-etal-2020-clue,
  abstract = {The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. Our benchmark is released at https://www.cluebenchmarks.com},
  address = {Barcelona, Spain (Online)},
  author = {Xu, Liang  and
Hu, Hai and
Zhang, Xuanwei and
Li, Lu and
Cao, Chenjie and
Li, Yudong and
Xu, Yechen and
Sun, Kai and
Yu, Dian and
Yu, Cong and
Tian, Yin and
Dong, Qianqian and
Liu, Weitang and
Shi, Bo and
Cui, Yiming and
Li, Junyi and
Zeng, Jun and
Wang, Rongzhao and
Xie, Weijian and
Li, Yanting and
Patterson, Yina and
Tian, Zuoyu and
Zhang, Yiwen and
Zhou, He and
Liu, Shaoweihua and
Zhao, Zhe and
Zhao, Qipeng and
Yue, Cong and
Zhang, Xinrui and
Yang, Zhengliang and
Richardson, Kyle and
Lan, Zhenzhong },
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
        prompt="Given an App description text, find the appropriate fine-grained category",
    )

    samples_per_label = 32

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["n_experiments"] = 5
        return metadata_dict


class MultilingualSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualSentiment",
        description="A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative",
        reference="https://github.com/tyqiangz/multilingual-sentiment-datasets",
        dataset={
            "path": "C-MTEB/MultilingualSentiment-classification",
            "revision": "46958b007a63fdbf239b7672c25d0bea67b5ea1a",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["validation", "test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
        prompt="Classify sentiment of the customer review into positive, neutral, or negative",
    )

    samples_per_label = 32


class JDReview(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JDReview",
        description="review for iphone",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "C-MTEB/JDReview-classification",
            "revision": "b7c64bd89eb87f8ded463478346f76731f07bf8b",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{xiao2023c,
  author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
  journal = {arXiv preprint arXiv:2309.07597},
  title = {C-pack: Packaged resources to advance general chinese embedding},
  year = {2023},
}
""",
        prompt="Classify the customer review for iPhone on e-commerce platform into positive or negative",
    )

    samples_per_label = 32


class OnlineShopping(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OnlineShopping",
        description="Sentiment Analysis of User Reviews on Online Shopping Websites",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "C-MTEB/OnlineShopping-classification",
            "revision": "e610f2ebd179a8fda30ae534c3878750a96db120",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{xiao2023c,
  author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
  journal = {arXiv preprint arXiv:2309.07597},
  title = {C-pack: Packaged resources to advance general chinese embedding},
  year = {2023},
}
""",
        prompt="Classify the customer review for online shopping into positive or negative",
    )

    samples_per_label = 32


class Waimai(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Waimai",
        description="Sentiment Analysis of user reviews on takeaway platforms",
        reference="https://aclanthology.org/2023.nodalida-1.20/",
        dataset={
            "path": "C-MTEB/waimai-classification",
            "revision": "339287def212450dcaa9df8c22bf93e9980c7023",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{xiao2023c,
  author = {Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighof, Niklas},
  journal = {arXiv preprint arXiv:2309.07597},
  title = {C-pack: Packaged resources to advance general chinese embedding},
  year = {2023},
}
""",
        prompt="Classify the customer review from a food takeaway platform into positive or negative",
    )

    samples_per_label = 32
