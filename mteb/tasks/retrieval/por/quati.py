from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class Quati(AbsTaskRetrieval):
    """Quati — Brazilian Portuguese native web retrieval, 50 topics over a 50k pool."""

    metadata = TaskMetadata(
        name="Quati",
        description=(
            "Quati: native Brazilian Portuguese web retrieval benchmark. 50 test "
            "topics with ~1,900 human relevance judgements over a Brazilian web "
            "passage pool (ClueWeb22-PT). First PT-BR retrieval benchmark designed "
            "without translation; queries are naturally-occurring Brazilian web "
            "questions. Evaluated over a 50k corpus subsample (all judged passages "
            "plus a fixed random sample of the 1M pool) for tractability."
        ),
        reference="https://aclanthology.org/2024.stil-1.19/",
        dataset={
            "path": "MTEB-BR/Quati",
            "revision": "396da597371d9e9490af0191f46b472693f576d2",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2024-06-30"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bueno-etal-2024-quati,
  title = {{Q}uati: A {B}razilian {P}ortuguese Information Retrieval Dataset from Native Speakers},
  author = {Bueno, Mirelle  and
      Maia Sanches, Eduardo  and
      Lotufo, Roberto  and
      Nogueira, Rodrigo},
  booktitle = {Proceedings of the 16th Symposium in Information and Human Language Technology (STIL 2024)},
  year = {2024},
  url = {https://aclanthology.org/2024.stil-1.19/},
}
""",
    )
