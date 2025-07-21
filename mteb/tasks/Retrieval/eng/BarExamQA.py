from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BarExamQA(AbsTaskRetrieval):
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
        task_subtypes=["Text Retrieval", "Question Answering"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""\
@inproceedings{Zheng_2025, series={CSLAW ’25},
   title={A Reasoning-Focused Legal Retrieval Benchmark},
   url={http://dx.doi.org/10.1145/3709025.3712219},
   DOI={10.1145/3709025.3712219},
   booktitle={Proceedings of the Symposium on Computer Science and Law on ZZZ},
   publisher={ACM},
   author={Zheng, Lucia and Guha, Neel and Arifov, Javokhir and Zhang, Sarah and Skreta, Michal and Manning, Christopher D. and Henderson, Peter and Ho, Daniel E.},
   year={2025},
   month=mar, pages={169–193},
   collection={CSLAW ’25},
   eprint={2505.03970}
}""",
    )
