from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class Robust04InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="Robust04InstructionRetrieval",
        description="Measuring retrieval instruction following ability on Robust04 narratives.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/robust04-instructions-old",
            "revision": "a5a1c4fe2bc528ac12e83f8cdf82178da85d2f1d",
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
        n_samples={"eng": 47544 * 2},
        avg_character_length={"eng": 2471.0398058252426},
    )
