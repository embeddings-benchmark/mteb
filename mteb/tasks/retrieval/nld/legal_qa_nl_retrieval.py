from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LegalQANLRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="LegalQANLRetrieval",
        description="To this end, we create and publish a Dutch legal QA dataset, consisting of question-answer pairs "
        "with attributions to Dutch law articles.",
        reference="https://aclanthology.org/2024.nllp-1.12/",
        dataset={
            "path": "clips/mteb-nl-legalqa-pr",
            "revision": "8f593522dfbe7ec07055ca9d38a700e7643d3882",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nld-Latn"],
        main_score="ndcg_at_10",
        date=("2021-05-01", "2021-08-26"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{redelaar2024attributed,
  author = {Redelaar, Felicia and Van Drie, Romy and Verberne, Suzan and De Boer, Maaike},
  booktitle = {Proceedings of the natural legal language processing workshop 2024},
  pages = {154--165},
  title = {Attributed Question Answering for Preconditions in the Dutch Law},
  year = {2024},
}
""",
        prompt={
            "query": "Gegeven een juridische vraag, haal documenten op die kunnen helpen bij het beantwoorden van de vraag"
        },
    )
