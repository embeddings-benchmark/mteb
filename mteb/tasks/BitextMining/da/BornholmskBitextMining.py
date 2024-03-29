from __future__ import annotations

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class BornholmBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="BornholmBitextMining",
        dataset={
            "path": "strombergnlp/bornholmsk_parallel",
            "revision": "3bc5cfb4ec514264fe2db5615fac9016f7251552",
        },
        description="Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.",
        reference="https://aclanthology.org/W19-6138/",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["da", "da-bornholm"],
        main_score="f1",
        date=None,
        form=None,
        domains=None,
        license=None,
        task_subtypes=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        avg_character_length={"test": 89.7},
        n_samples={"test": 500},
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("da", "sentence1")
        self.dataset = self.dataset.rename_column("da_bornholm", "sentence2")
