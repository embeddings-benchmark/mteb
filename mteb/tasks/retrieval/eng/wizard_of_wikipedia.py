from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WiardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WiardOfWikipedia",
        description="WiardOfWikipedia",
        reference=None,
        dataset={
            "path": "DeepPavlov/wizard_of_wikipedia",
            "revision": "a806e8f492e91cdcfe5a86ff6fa5cefaf2dcf11c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{elgohary-etal-2019-unpack,
    title = "Can You Unpack That? Learning to Rewrite Questions-in-Context",
    author = "Elgohary, Ahmed  and
      Peskov, Denis  and
      Boyd-Graber, Jordan",
    editor = "Inui, Kentaro  and
      Jiang, Jing  and
      Ng, Vincent  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1605/",
    doi = "10.18653/v1/D19-1605",
    pages = "5918--5924",
}
""",
    )
