from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class WebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="WebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://mcgill-nlp.github.io/weblinx",
        dataset={
            "path": "McGill-NLP/WebLINX",
            "name": "reranking",
            "revision": "f3c18563a49fa8ef4559eb8f2b2e2c9845f71bf8",
        },
        type="Reranking",
        category="p2p",
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["eng-Latn"],
        main_score="recall_at_10",
        date=("2023-03-01", "2023-10-30"),
        form=["written"],
        domains=["Academic", "Web"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="CC BY-NC-SA 4.0",
        socioeconomic_status="mixed",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="created",
        bibtex_citation="""
@misc{lù2024weblinx,
      title={WebLINX: Real-World Website Navigation with Multi-Turn Dialogue}, 
      author={Xing Han Lù and Zdeněk Kasner and Siva Reddy},
      year={2024},
      eprint={2402.05930},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
        """,
        n_samples={
            "validation": 1301,
            "test_iid": 1438,
            "test_cat": 3560,
            "test_web": 3144,
            "test_vis": 5298,
            "test_geo": 4916,
        },
        avg_character_length={
            "validation": 1647.52,
            "test_iid": 1722.63,
            "test_cat": 2149.66,
            "test_web": 1831.46,
            "test_vis": 1737.26,
            "test_geo": 1742.66,
        },
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self._datasets = {}

        for split in self.metadata.eval_splits:
            self._datasets[split] = datasets.load_dataset(
                split=split, **self.metadata_dict["dataset"]
            )

        self.dataset = datasets.DatasetDict(
            {split: self._datasets[split] for split in self.metadata.eval_splits}
        )

        self.dataset_transform()

        self.data_loaded = True
