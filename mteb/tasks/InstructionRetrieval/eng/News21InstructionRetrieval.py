from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class News21InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="News21InstructionRetrieval",
        description="Measuring retrieval instruction following ability on News21 narratives.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/news21-instructions",
            "revision": "374e11e7d416b2076f983d6b4af0a2389225f5e8",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        form=["written"],
        domains=["News"],
        task_subtypes=None,
        license="MIT",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@misc{weller2024followir,
      title={FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions}, 
      author={Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      year={2024},
      eprint={2403.15246},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples=None,
        avg_character_length=None,
    )
