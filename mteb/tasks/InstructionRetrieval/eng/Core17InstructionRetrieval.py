from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskInstructionRetrieval import AbsTaskInstructionRetrieval


class Core17InstructionRetrieval(AbsTaskInstructionRetrieval):
    metadata = TaskMetadata(
        name="Core17InstructionRetrieval",
        description="Measuring retrieval instruction following ability on Core17 narratives for the FollowIR benchmark.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/core17-instructions",
            "revision": "e39ff896cf3efbbdeeb950e6bd7c79f266995b07",
        },
        type="InstructionRetrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{weller2024followir,
      title={FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions}, 
      author={Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
      year={2024},
      eprint={2403.15246},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        descriptive_stats={
            "n_samples": {"eng": 19919 * 2},
            "test": {
                "num_docs": 19899,
                "num_queries": 20,
                "average_document_length": 2233.0329664807277,
                "average_query_length": 109.75,
                "average_instruction_length": 295.55,
                "average_changed_instruction_length": 355.2,
                "average_relevant_docs_per_query": 32.7,
                "average_top_ranked_per_query": 1000.0,
            },
        },
    )
