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
            "revision": "e0144086b45fe31ac125e9ac1a83b6a409bb6ca6",
        },
        type="InstructionRetrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        form=["written"],
        domains=["News"],
        task_subtypes=[],
        license="MIT",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@misc{weller2024followir,
      title={FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions}, 
      author={Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      year={2024},
      eprint={2403.15246},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples={"eng": 30953 * 2},
        avg_character_length={"eng": 2983.724665391969},
    )
