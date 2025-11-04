from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class XLWICNLPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="XLWICNLPairClassification",
        description="The Word-in-Context dataset (WiC) addresses the dependence on sense inventories by reformulating "
        "the standard disambiguation task as a binary classification problem; but, it is limited to the "
        "English language. We put forward a large multilingual benchmark, XL-WiC, featuring gold standards "
        "in 12 new languages from varied language families and with different degrees of resource "
        "availability, opening room for evaluation scenarios such as zero-shot cross-lingual transfer. ",
        reference="https://aclanthology.org/2020.emnlp-main.584.pdf",
        dataset={
            "path": "clips/mteb-nl-xlwic",
            "revision": "0b33ce358b1b5d500ff3715ba3d777b4d2c21cb0",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        date=("2019-10-04", "2019-10-04"),
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="max_ap",
        domains=["Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{raganato2020xl,
  author = {Raganato, A and Pasini, T and Camacho-Collados, J and Pilehvar, M and others},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  organization = {Association for Computational Linguistics (ACL)},
  pages = {7193--7206},
  title = {XL-WiC: A multilingual benchmark for evaluating semantic contextualization},
  year = {2020},
}
""",
        prompt={
            "query": "Zoek tekst die semantisch vergelijkbaar is met de gegeven tekst."
        },
    )
