from __future__ import annotations

import datasets

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
        eval_langs=["nb", "nn"],
        main_score="f1",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples={"test": 2050},
        avg_character_length={"test": 1884.0},
    )

    def dataset_transform(self):
        # Convert to standard format
        self.dataset = self.dataset.rename_column("nb", "sentence1")
        self.dataset = self.dataset.rename_column("nn", "sentence2")
