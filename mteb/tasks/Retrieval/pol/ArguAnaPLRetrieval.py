from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaPL(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAna-PL",
        description="ArguAna-PL",
        reference="https://huggingface.co/datasets/clarin-knext/arguana-pl",
        dataset={
            "path": "mteb/ArguAna-PL",
            "revision": "0019de73979daf75f6526a0fc5ef77ebce27b8f9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Written"],
        task_subtypes=None,
        license="cc-by-sa-4.0",
        annotations_creators=None,
        dialect=[],
        sample_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        adapted_from=["ArguAna"],
    )
