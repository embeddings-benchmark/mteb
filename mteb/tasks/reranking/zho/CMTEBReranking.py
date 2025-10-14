from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class T2Reranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="T2Reranking",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="https://arxiv.org/abs/2304.03679",
        dataset={
            "path": "mteb/T2Reranking",
            "revision": "a34fe6bc0dff185af1228e49a0f6fb1de1565627",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=None,
        domains=[],
        task_subtypes=None,
        license="not specified",
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        prompt="Given a Chinese search query, retrieve web passages that answer the question",
        bibtex_citation=r"""
@misc{xie2023t2ranking,
  archiveprefix = {arXiv},
  author = {Xiaohui Xie and Qian Dong and Bingning Wang and Feiyang Lv and Ting Yao and Weinan Gan and Zhijing Wu and Xiangsheng Li and Haitao Li and Yiqun Liu and Jin Ma},
  eprint = {2304.03679},
  primaryclass = {cs.IR},
  title = {T2Ranking: A large-scale Chinese Benchmark for Passage Ranking},
  year = {2023},
}
""",
    )


class MMarcoReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MMarcoReranking",
        description="mMARCO is a multilingual version of the MS MARCO passage ranking dataset",
        reference="https://github.com/unicamp-dl/mMARCO",
        dataset={
            "path": "mteb/MMarcoReranking",
            "revision": "91c0f9d5cbd4cdc7ecd7039c5c2e57740cb62b00",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        prompt="Given a Chinese search query, retrieve web passages that answer the question",
        bibtex_citation=r"""
@misc{bonifacio2021mmarco,
  archiveprefix = {arXiv},
  author = {Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
  eprint = {2108.13897},
  primaryclass = {cs.CL},
  title = {mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset},
  year = {2021},
}
""",
    )


class CMedQAv1(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CMedQAv1-reranking",
        description="Chinese community medical question answering",
        prompt="Given a Chinese community medical question, retrieve replies that best answer the question",
        reference="https://github.com/zhangsheng93/cMedQA",
        dataset={
            "path": "mteb/CMedQAv1-reranking",
            "revision": "f77f0345c9b6ac0a2447a3a71483d5b27393cb35",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=("2017-01-01", "2017-07-26"),
        domains=["Medical", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{zhang2017chinese,
  author = {Zhang, Sheng and Zhang, Xin and Wang, Hui and Cheng, Jiajun and Li, Pei and Ding, Zhaoyun},
  journal = {Applied Sciences},
  number = {8},
  pages = {767},
  publisher = {Multidisciplinary Digital Publishing Institute},
  title = {Chinese Medical Question Answer Matching Using End-to-End Character-Level Multi-Scale CNNs},
  volume = {7},
  year = {2017},
}
""",
    )


class CMedQAv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CMedQAv2-reranking",
        description="Chinese community medical question answering",
        prompt="Given a Chinese community medical question, retrieve replies that best answer the question",
        reference="https://github.com/zhangsheng93/cMedQA2",
        dataset={
            "path": "mteb/CMedQAv2-reranking",
            "revision": "01399b6956a4c6bf9edc5b6b90e176aeeb399686",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["cmn-Hans"],
        main_score="map_at_1000",
        date=None,
        domains=["Medical", "Written"],
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@article{8548603,
  author = {S. Zhang and X. Zhang and H. Wang and L. Guo and S. Liu},
  doi = {10.1109/ACCESS.2018.2883637},
  issn = {2169-3536},
  journal = {IEEE Access},
  keywords = {Biomedical imaging;Data mining;Semantics;Medical services;Feature extraction;Knowledge discovery;Medical question answering;interactive attention;deep learning;deep neural networks},
  month = {},
  number = {},
  pages = {74061-74071},
  title = {Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection},
  volume = {6},
  year = {2018},
}
""",
    )
