from __future__ import annotations

import random

import datasets

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata

TEST_SAMPLES = 2048


class VieMedEVBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="VieMedEVBitextMining",
        dataset={
            "path": "nhuvo/MedEV",
            "revision": "d03c69413bc53d1cea5a5375b3a953c4fee35ecd",
            "trust_remote_code": True,
        },
        description="A high-quality Vietnamese-English parallel data from the medical domain for machine translation",
        reference="https://aclanthology.org/2015.iwslt-evaluation.11/",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn", "vie-Latn"],
        main_score="f1",
        date=("2024-08-28", "2022-03-28"),
        form=["written"],
        domains=["Medical"],
        task_subtypes=[],
        license="cc-by-nc",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="human-translated and localized",
        bibtex_citation="""@inproceedings{medev,
    title     = {{Improving Vietnamese-English Medical Machine Translation}},
    author    = {Nhu Vo and Dat Quoc Nguyen and Dung D. Le and Massimo Piccardi and Wray Buntine},
    booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING)},
    year      = {2024}
}""",
        n_samples={"test": TEST_SAMPLES},
        avg_character_length={"test": 139.23},
    )

    def dataset_transform(self):
        # Convert to standard format
        ds = {}
        seed = 42
        random.seed(seed)
        # Get all texts
        all_texts = self.dataset["test"]["text"]

        # Determine the midpoint of the list
        mid_index = len(all_texts) // 2
        # Pairs are in two halves
        en_sentences = all_texts[:mid_index]
        vie_sentences = all_texts[mid_index:]
        assert len(en_sentences) == len(
            vie_sentences
        ), "The split does not result in equal halves."

        # Downsample
        indices = list(range(len(en_sentences)))
        random.shuffle(indices)
        sample_indices = indices[:TEST_SAMPLES]
        en_sentences = [en_sentences[i] for i in sample_indices]
        vie_sentences = [vie_sentences[i] for i in sample_indices]
        assert (
            len(en_sentences) == len(vie_sentences) == TEST_SAMPLES
        ), f"Exceeded {TEST_SAMPLES} samples for 'test' split."

        # Return dataset
        ds["test"] = datasets.Dataset.from_dict(
            {"sentence1": vie_sentences, "sentence2": en_sentences}
        )
        self.dataset = datasets.DatasetDict(ds)
