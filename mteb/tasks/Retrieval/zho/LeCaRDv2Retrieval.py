from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LeCaRDv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LeCaRDv2",
        description="The task involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries.",
        reference="https://github.com/THUIR/LeCaRDv2",
        dataset={
            "path": "mteb/LeCaRDv2",
            "revision": "b78e18688c3d012a33dc3676597c1d1b2243ce1c",
        },
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["zho-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation="""@misc{li2023lecardv2,
      title={LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset},
      author={Haitao Li and Yunqiu Shao and Yueyue Wu and Qingyao Ai and Yixiao Ma and Yiqun Liu},
      year={2023},
      eprint={2310.17609},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
    )
