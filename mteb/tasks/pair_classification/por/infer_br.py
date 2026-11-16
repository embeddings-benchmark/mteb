from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class InferBR(AbsTaskPairClassification):
    """InferBR — Brazilian Portuguese NLI, binarized as Entailment vs other."""

    metadata = TaskMetadata(
        name="InferBR",
        description=(
            "InferBR: Brazilian Portuguese Natural Language Inference. 10,528 "
            "premise-hypothesis pairs annotated as CONTRADICTION / ENTAILMENT / "
            "NEUTRAL, with quality-flagged samples removed from the original "
            "release. For PairClassification we binarize: ENTAILMENT (label=1) "
            "vs other (label=0). Complements AssinRTE with a larger, modern "
            "sample pool."
        ),
        reference="https://aclanthology.org/2024.lrec-main.793/",
        dataset={
            "path": "MTEB-BR/InferBR",
            "revision": "3228f11e8d31191d303871c19e67e0a2bc5bc862",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="max_ap",
        date=("2023-01-01", "2024-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bencke-etal-2024-inferbr,
  title = {{I}nfer{BR}: A Natural Language Inference Dataset in {P}ortuguese},
  author = {Bencke, Luciana  and
              Pereira, Francielle Vasconcellos  and
              Santos, Moniele Kunrath  and
              Moreira, Viviane},
  editor = {Calzolari, Nicoletta  and
              Kan, Min-Yen  and
              Hoste, Veronique  and
              Lenci, Alessandro  and
              Sakti, Sakriani  and
              Xue, Nianwen},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  month = may,
  year = {2024},
  address = {Torino, Italia},
  publisher = {ELRA and ICCL},
  pages = {9050--9060},
  url = {https://aclanthology.org/2024.lrec-main.793/},
}
""",
    )
