from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class Core17InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="Core17InstructionRetrieval",
        description="Measuring retrieval instruction following ability on Core17 narratives.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/core17-instructions",
            "revision": "e783a88ec6bc4bbdc1ba998edd85edeaf3d820f7",
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
        n_samples={"eng": 19735 * 2},
        avg_character_length={"eng": 2747.2883966244726},
    )
