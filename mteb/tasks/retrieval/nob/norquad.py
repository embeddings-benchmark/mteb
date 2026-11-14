from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NorQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NorQuadRetrieval",
        dataset={
            "path": "mteb/NorQuadRetrieval",
            "revision": "3960e9daeab87d09d72c5547745e9909cdd75549",
        },
        description="Human-created question for Norwegian wikipedia passages.",
        reference="https://aclanthology.org/2023.nodalida-1.17/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2023-12-31"),
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{ivanova-etal-2023-norquad,
  address = {T{\'o}rshavn, Faroe Islands},
  archiveprefix = {arXiv},
  author = {Ivanova, Sardana  and
Andreassen, Fredrik  and
Jentoft, Matias  and
Wold, Sondre  and
{\O}vrelid, Lilja},
  booktitle = {Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Alum{\"a}e, Tanel  and
Fishel, Mark},
  eprint = {2305.01957},
  month = may,
  pages = {159--168},
  publisher = {University of Tartu Library},
  title = {{N}or{Q}u{AD}: {N}orwegian Question Answering Dataset},
  url = {https://aclanthology.org/2023.nodalida-1.17},
  year = {2023},
}
""",
        prompt={
            "query": "Given a question in Norwegian, retrieve the answer from Wikipedia articles"
        },
    )
