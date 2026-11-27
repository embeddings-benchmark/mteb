from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class AssinRTE(AbsTaskPairClassification):
    """ASSIN v1 RTE binarized: None=0, Entailment+Paraphrase=1. Brazilian subset."""

    metadata = TaskMetadata(
        name="AssinRTE",
        description=(
            "ASSIN (Avaliação de Similaridade Semântica e Inferência Textual) v1, "
            "Brazilian Portuguese subset. Recognizing Textual Entailment task: "
            "given a premise-hypothesis pair, predict whether the hypothesis "
            "is entailed by the premise. Labels binarized as {None: 0, "
            "Entailment+Paraphrase: 1} following Assin2RTE convention."
        ),
        reference="https://aclanthology.org/2020.lrec-1.319/",  # ASSIN follow-up survey
        dataset={
            "path": "MTEB-BR/AssinRTE",
            "revision": "ae4452c01f94260a77a6c6a2fe09c6ff41d19ddd",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="max_ap",
        date=("2016-01-01", "2016-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=["Textual Entailment"],
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
  url = {http://propor2016.di.fc.ul.pt/},
}
""",
    )
