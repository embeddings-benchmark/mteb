from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MedPTRetrieval(AbsTaskRetrieval):
    """MedPT — Brazilian Portuguese biomedical QA retrieval, 500 queries × 500 docs."""

    metadata = TaskMetadata(
        name="MedPTRetrieval",
        description=(
            "MedPT: Brazilian Portuguese biomedical question-answer retrieval. "
            "Subsample of 500 question/answer pairs from the AKCIT/MedPT corpus "
            "(384k QA pairs across 3,288 medical conditions and 1,278 "
            "specialties), stratified across 7 question_type categories. Each "
            "question has one canonical answer as its single relevant document. "
            "Tests biomedical-domain retrieval transfer for PT-BR embedding models."
        ),
        reference="https://huggingface.co/datasets/AKCIT/MedPT",
        dataset={
            "path": "MTEB-BR/MedPTRetrieval",
            "revision": "d54b959c65dd4af380386bf91c9ad05753add8e6",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{farber-etal-2026-medpt,
  title = {{M}ed{PT}: A Massive Medical Question Answering Dataset for {B}razilian-{P}ortuguese Speakers},
  author = {Farber, Fernanda Bufon and Brito, Iago Alves and Dollis, Julia Soares and Ribeiro, Pedro Schindler Freire Brasil and Sousa, Rafael Teixeira and Galv{\~a}o Filho, Arlindo Rodrigues},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = may,
  year = {2026},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  pages = {2419--2429},
  doi = {10.63317/5mrjjtbsf6rv},
  url = {https://arxiv.org/abs/2511.11878},
}
""",
    )
