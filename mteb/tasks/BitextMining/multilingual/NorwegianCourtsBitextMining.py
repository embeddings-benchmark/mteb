from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["nob-Latn", "nno-Latn"],
        main_score="f1",
        date=("2020-01-01", "2020-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{opus4,
  title={OPUS-MT — Building open translation services for the World},
  author={Tiedemann, J{\"o}rg and Thottingal, Santhosh},
  booktitle={Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)},
  year={2020}
}
""",
        prompt="Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
