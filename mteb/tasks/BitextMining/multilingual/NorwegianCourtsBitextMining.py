from __future__ import annotations

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.TaskMetadata import TaskMetadata


class NorwegianCourtsBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="NorwegianCourtsBitextMining",
        dataset={
            "path": "kardosdrur/norwegian-courts",
            "revision": "d79af07e969a6678fcbbe819956840425816468f",
        },
        description="Nynorsk and Bokmål parallel corpus from Norwegian courts. Norwegian courts have two standardised written languages. Bokmål is a variant closer to Danish, while Nynorsk was created to resemble regional dialects of Norwegian.",
        reference="https://opus.nlpl.eu/index.php",
        type="BitextMining",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["nob-Latn", "nno-Latn"],
        main_score="f1",
        date=("2020-01-01", "2020-12-31"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="CC BY 4.0",
        socioeconomic_status="mixed",
        annotations_creators="human-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
@inproceedings{opus4,
  title={OPUS-MT — Building open translation services for the World},
  author={Tiedemann, J{\"o}rg and Thottingal, Santhosh},
  booktitle={Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)},
  year={2020}
}
""",
        n_samples={"test": 2050},
        avg_character_length={"test": 1884.0},
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
