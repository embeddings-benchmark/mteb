from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CitationPredictionLegalBenchPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CitationPredictionLegalBenchPC",
        description="Citation Prediction Classification LegalBench Dataset",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "citation_prediction_classification",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }""",
        n_samples={"test": 108},
        avg_character_length={"test": 188.95},
    )

    def dataset_transform(self):
        _dataset = {}
        mapping = {"yes": 1, "no": 0}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]
            hf_dataset = hf_dataset.map(
                lambda example: {
                    "answer": mapping.get(example["answer"].lower(), example["answer"])
                }
            )

            _dataset[split] = [
                {
                    "sent1": hf_dataset["text"],
                    "sent2": hf_dataset["citation"],
                    "labels": hf_dataset["answer"],
                }
            ]
        self.dataset = _dataset
