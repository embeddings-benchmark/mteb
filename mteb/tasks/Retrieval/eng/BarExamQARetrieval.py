from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class BarExamQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        dataset={
            "path": "isaacus/mteb-barexam-qa",
            "revision": "49c07b6",
        },
        name="BarExamQA",
        description="A benchmark for retrieving legal provisions that answer US bar exam questions.",
        reference="https://huggingface.co/datasets/reglab/barexam_qa",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2024-08-14", "2025-07-18"),
        domains=["Legal", "Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Zheng_2025,
  author = {Zheng, Lucia and Guha, Neel and Arifov, Javokhir and Zhang, Sarah and Skreta, Michal and Manning, Christopher D. and Henderson, Peter and Ho, Daniel E.},
  booktitle = {Proceedings of the Symposium on Computer Science and Law on ZZZ},
  collection = {CSLAW ’25},
  doi = {10.1145/3709025.3712219},
  eprint = {2505.03970},
  month = mar,
  pages = {169–193},
  publisher = {ACM},
  series = {CSLAW ’25},
  title = {A Reasoning-Focused Legal Retrieval Benchmark},
  url = {http://dx.doi.org/10.1145/3709025.3712219},
  year = {2025},
}
""",
    )
