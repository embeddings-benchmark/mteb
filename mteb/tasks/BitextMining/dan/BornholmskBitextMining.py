from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class BornholmBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="BornholmBitextMining",
        dataset={
            "path": "strombergnlp/bornholmsk_parallel",
            "revision": "3bc5cfb4ec514264fe2db5615fac9016f7251552",
            "trust_remote_code": True,
        },
        description="Danish Bornholmsk Parallel Corpus. Bornholmsk is a Danish dialect spoken on the island of Bornholm, Denmark. Historically it is a part of east Danish which was also spoken in Scania and Halland, Sweden.",
        reference="https://aclanthology.org/W19-6138/",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="f1",
        date=("2019-01-01", "2019-12-31"),
        domains=["Web", "Social", "Fiction", "Written"],
        license="cc-by-4.0",
        task_subtypes=["Dialect pairing"],
        annotations_creators="expert-annotated",
        dialect=["da-dan-bornholm"],
        sample_creation="created",
        bibtex_citation="""
@inproceedings{derczynskiBornholmskNaturalLanguage2019,
	title = {Bornholmsk natural language processing: Resources and tools},
	url = {https://pure.itu.dk/ws/files/84551091/W19_6138.pdf},
	shorttitle = {Bornholmsk natural language processing},
	pages = {338--344},
	booktitle = {Proceedings of the Nordic Conference of Computational Linguistics (2019)},
	publisher = {Linköping University Electronic Press},
	author = {Derczynski, Leon and Kjeldsen, Alex Speed},
	urldate = {2024-04-24},
	date = {2019},
	file = {Available Version (via Google Scholar):/Users/au554730/Zotero/storage/FBQ73ZYN/Derczynski and Kjeldsen - 2019 - Bornholmsk natural language processing Resources .pdf:application/pdf},
}
""",
        descriptive_stats={
            "n_samples": {"test": 500},
            "test": {
                "average_sentence1_length": 49.834,
                "average_sentence2_length": 38.888,
                "num_samples": 500,
            },
        },
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("da", "sentence1")
        self.dataset = self.dataset.rename_column("da_bornholm", "sentence2")
