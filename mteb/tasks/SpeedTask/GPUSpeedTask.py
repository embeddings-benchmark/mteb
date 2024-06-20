from __future__ import annotations

from mteb.abstasks.AbsTaskSpeedTask import AbsTaskSpeedTask
from mteb.abstasks.TaskMetadata import TaskMetadata


class GPUSpeedTask(AbsTaskSpeedTask):
    device = "cuda"
    metadata = TaskMetadata(
        name="GPUSpeedTask",
        description="Time taken to encode the text 'The Ugly Duckling' split by paragraphs on a GPU.",
        reference="https://github.com/KennethEnevoldsen/scandinavian-embedding-benchmark/blob/c8376f967d1294419be1d3eb41217d04cd3a65d3/src/seb/registered_tasks/speed.py#L83-L96",
        dataset={"path": " ", "revision": "1.0"},
        type="Speed",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="avg_words_per_sec",
        date=("2024-06-20", "2024-06-20"),
        form=["written"],
        domains=["Fiction"],
        task_subtypes=[],
        license="Not specified",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="",
        n_samples={"test": 1},
        avg_character_length={"test": 3591},
    )
