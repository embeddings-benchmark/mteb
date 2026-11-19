from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class AssinSTS(AbsTaskSTS):
    """ASSIN v1 STS, Brazilian subset, relatedness 1-5."""

    metadata = TaskMetadata(
        name="AssinSTS",
        description=(
            "ASSIN (Avaliação de Similaridade Semântica e Inferência Textual) v1, "
            "Brazilian Portuguese subset. Semantic textual similarity task: each "
            "pair of sentences is annotated with a relatedness score on a "
            "1 (no similarity) to 5 (paraphrase) Likert scale by human experts."
        ),
        reference="https://aclanthology.org/2020.lrec-1.319/",
        dataset={
            "path": "MTEB-BR/AssinSTS",
            "revision": "8215a4634017a35f623df8ad18cea319f6a74f34",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{fonseca-etal-2016-assin,
  title = {{ASSIN}: {A}valia{\c{c}}{\~a}o de {S}imilaridade {S}em{\^a}ntica e {I}nfer{\^e}ncia {T}extual},
  author = {Fonseca, Erick  and
      Santos, Leandro  and
      Criscuolo, Marcelo  and
      Alu{\'\i}sio, Sandra},
  booktitle = {Proceedings of the 12th International Conference on Computational Processing of the Portuguese Language},
  year = {2016},
}
""",
    )
    min_score = 1
    max_score = 5
