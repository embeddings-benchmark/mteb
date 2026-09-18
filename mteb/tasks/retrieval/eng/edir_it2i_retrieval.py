from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EDIRIT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EDIRIT2IRetrieval",
        description=(
            "EDIR evaluates composed image retrieval: retrieve an edited image from "
            "a reference image and a textual modification instruction. This task uses "
            "the public streamlined release, whose corpus removes the paper's extra "
            "distractor images."
        ),
        reference="https://aclanthology.org/2026.acl-long.2144/",
        dataset={
            "path": "whybe-choi/EDIRIT2IRetrieval",
            "revision": "6b371e3b3c37fffeb02bf029fc9beba3515ee7ce",
        },
        type="Any2AnyRetrieval",
        category="it2i",
        modalities=["image", "text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2026-01-22", "2026-01-22"),
        domains=["Scene", "Constructed"],
        task_subtypes=["Image Text Retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation=r"""
@inproceedings{song-etal-2026-rethinking-composed,
  author = {Song, Tingyu and Zhang, Yanzhao and Li, Mingxin and Guo, Zhuoning and Long, Dingkun and Xie, Pengjun and Zhang, Siyue and Zhao, Yilun and Wu, Shu},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {46224--46242},
  title = {Rethinking Composed Image Retrieval Evaluation: A Fine-Grained Benchmark from Image Editing},
  url = {https://aclanthology.org/2026.acl-long.2144/},
  year = {2026},
}
""",
        prompt={
            "query": "Given an image, find a similar image satisfying the query.",
        },
    )
