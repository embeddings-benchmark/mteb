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
            "revision": "ed1c933c2b3617e5700d8a7ebe07f5975969a453",
        },
        type="Reranking",
        category="p2p",
        modalities=["text"],
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["eng-Latn"],
        main_score="mrr",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
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
