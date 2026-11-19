from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

# Pinned to the SHA observed during 2026-05-23 preflight. Refresh via:
#   from huggingface_hub import HfApi
#   HfApi().dataset_info("franciellevargas/HateBR").sha


class HateBR(AbsTaskClassification):
    """HateBR binary offensive-language classification."""

    metadata = TaskMetadata(
        name="HateBR",
        description=(
            "HateBR: Brazilian Portuguese Instagram comments annotated for "
            "offensive language and hate speech. Binary classification of "
            "offensive vs non-offensive comments. 7,000 comments, expert-"
            "annotated by linguists with kappa agreement >0.83."
        ),
        reference="https://aclanthology.org/2022.lrec-1.777/",
        dataset={
            "path": "MTEB-BR/HateBR",
            "revision": "38d266a46743bed3ce28bfa914415c1ac66841c7",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        date=("2018-01-01", "2021-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{vargas-etal-2022-hatebr,
  title = {{H}ate{BR}: A Large Expert Annotated Corpus of {B}razilian {I}nstagram Comments for Offensive Language and Hate Speech Detection},
  author = {Vargas, Francielle  and
      Carvalho, Isabelle  and
      Rodrigues de G{\'o}es, Fabiana  and
      Pardo, Thiago  and
      Benevenuto, Fabr{\'\i}cio},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  month = jun,
  year = {2022},
  address = {Marseille, France},
  publisher = {European Language Resources Association},
  url = {https://aclanthology.org/2022.lrec-1.777},
  pages = {7174--7183},
}
""",
    )

    # Real HateBR CSV columns (verified 2026-05-23 preflight):
    # id, comentario, annotator1, annotator2, annotator3, label_final, links_post, account_post
    # `label_final` is the canonical binary label (0/1, balanced 3500/3500).
    input_column_name = "comentario"
    label_column_name = "label_final"
    samples_per_label = 32
