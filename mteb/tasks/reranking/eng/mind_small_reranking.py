import logging

from mteb._evaluators.retrieval_metrics import max_over_subqueries
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

logger = logging.getLogger(__name__)


class MindSmallReranking(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MindSmallReranking",
        description="Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        dataset={
            "path": "mteb/MindSmallReranking",
            "revision": "227478e3235572039f4f7661840e059f31ef6eb1",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_over_subqueries_map_at_1000",
        date=("2019-10-12", "2019-11-22"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve relevant news articles based on user browsing history",
        bibtex_citation=r"""
@inproceedings{wu-etal-2020-mind,
  address = {Online},
  author = {Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi,
Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.331},
  editor = {Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel},
  month = jul,
  pages = {3597--3606},
  publisher = {Association for Computational Linguistics},
  title = {{MIND}: A Large-scale Dataset for News
Recommendation},
  url = {https://aclanthology.org/2020.acl-main.331},
  year = {2020},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return max_over_subqueries(qrels, results, self.k_values)
