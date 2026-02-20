from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HagridRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HagridRetrieval",
        dataset={
            "path": "mteb/HagridRetrieval",
            "revision": "ae4f8bebcb82af2028863b778e1eebf4f5f23628",
        },
        reference="https://github.com/project-miracl/hagrid",
        description=(
            "HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset)"
            + "is a dataset for generative information-seeking scenarios. It consists of queries"
            + "along with a set of manually labelled relevant passages"
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-02-01", "2022-10-18"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{hagrid,
  author = {Ehsan Kamalloo and Aref Jafari and Xinyu Zhang and Nandan Thakur and Jimmy Lin},
  journal = {arXiv:2307.16883},
  title = {{HAGRID}: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution},
  year = {2023},
}
""",
    )
