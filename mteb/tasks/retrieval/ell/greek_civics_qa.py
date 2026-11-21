from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class GreekCivicsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="GreekCivicsQA",
        description="GreekCivicsQA is a set of 407 question and answer pairs related to Greek high school civics courses (Κοινωνική και Πολιτική Αγωγή), mined from Greek secondary education textbooks.",
        reference="https://huggingface.co/datasets/ilsp/greek_civics_qa",
        dataset={
            "path": "mteb/GreekCivicsQA",
            "revision": "61a2878c953e6a2bba9e073def24052739508aa1",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["default"],
        eval_langs=["ell-Grek"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2024-04-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
