from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class JaQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaQuADRetrieval",
        dataset={
            "path": "mteb/JaQuADRetrieval",
            "revision": "5713cd1b76a22b79c3595827c717c72b6cab8852",
        },
        description="Human-annotated question-answer pairs for Japanese wikipedia pages.",
        reference="https://arxiv.org/abs/2202.01764",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2022-01-01", "2022-12-31"),  # approximate guess
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-3.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{so2022jaquad,
  archiveprefix = {arXiv},
  author = {ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
  eprint = {2202.01764},
  primaryclass = {cs.CL},
  title = {{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
  year = {2022},
}
""",
    )
