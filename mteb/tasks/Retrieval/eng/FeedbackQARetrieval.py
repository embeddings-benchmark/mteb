from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class FeedbackQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FeedbackQARetrieval",
        description="Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment",
        reference="https://arxiv.org/abs/2204.03025",
        dataset={
            "path": "lt2c/fqa",
            "revision": "1ee1cd0",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="precision_at_1",
        date=("2020-01-01", "2022-04-01"),
        form=["written"],
        domains=["Web", "Government", "Medical"],
        task_subtypes=["Question answering"],
        license="Apache-2.0",
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=[],
        text_creation="created",
        bibtex_citation="""
@inproceedings{li-etal-2022-using,
    title = "Using Interactive Feedback to Improve the Accuracy and Explainability of Question Answering Systems Post-Deployment",
    author = "Li, Zichao  and
      Sharma, Prakhar  and
      Lu, Xing Han  and
      Cheung, Jackie  and
      Reddy, Siva",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.75",
    doi = "10.18653/v1/2022.findings-acl.75",
    pages = "926--937"
}
""",
        n_samples={"test": 1992},
        avg_character_length={"test": 1175}
    )
