from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FaithDialRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FaithDial",
        dataset={
            "path": "mteb/FaithDial",
            "revision": "303f0593746993a53c0c2ec976fb576369f6c9f3",
        },
        reference="https://mcgill-nlp.github.io/FaithDial",
        description=(
            "FaithDial is a faithful knowledge-grounded dialogue benchmark."
            + "It was curated by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). "
            + "It consists of conversation histories along with manually labelled relevant passage. "
            + "For the purpose of retrieval, we only consider the instances marked as 'Edification' in the VRM field, "
            + "as the gold passage associated with these instances is non-ambiguous."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-03-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dziri2022faithdial,
  author = {Dziri, Nouha and Kamalloo, Ehsan and Milton, Sivan and Zaiane, Osmar and Yu, Mo and Ponti, Edoardo M and Reddy, Siva},
  doi = {10.1162/tacl_a_00529},
  journal = {Transactions of the Association for Computational Linguistics},
  month = {12},
  pages = {1473--1490},
  publisher = {MIT Press},
  title = {{FaithDial: A Faithful Benchmark for Information-Seeking Dialogue}},
  volume = {10},
  year = {2022},
}
""",
    )
