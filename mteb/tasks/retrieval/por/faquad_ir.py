from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FaQuADIR(AbsTaskRetrieval):
    """FaQuAD-IR — Brazilian Portuguese academic retrieval, 900 queries / 244 paragraphs."""

    metadata = TaskMetadata(
        name="FaQuADIR",
        description=(
            "FaQuAD reformulated as PT-BR academic retrieval: given a question "
            "about Brazilian higher education, retrieve the source paragraph "
            "containing the answer. 900 questions over 244 unique paragraphs, "
            "drawn from 18 official documents of a federal-university CS "
            "program plus 21 Wikipedia articles about Brazil's higher-education "
            "system. Complements legal/web retrieval domains with academic/"
            "educational content."
        ),
        reference="https://github.com/liafacom/faquad",
        dataset={
            "path": "MTEB-BR/FaQuADIR",
            "revision": "a26eb12464c45a87afe686d6ec226657d9e1531e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Academic", "Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=["brazilian"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{sayama-etal-2019-faquad,
  author = {Sayama, H{\'e}lio Fonseca and Araujo, Anderson Vi{\c{c}}oso and Fernandes, Eraldo Rezende},
  booktitle = {2019 8th Brazilian Conference on Intelligent Systems (BRACIS)},
  title = {{F}a{Q}u{AD}: Reading Comprehension Dataset in the Domain of {B}razilian Higher Education},
  year = {2019},
  pages = {443--448},
  doi = {10.1109/BRACIS.2019.00084},
}
""",
    )
