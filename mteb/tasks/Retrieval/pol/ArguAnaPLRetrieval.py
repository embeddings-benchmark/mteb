from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class ArguAnaPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ArguAna-PL",
        description="ArguAna-PL",
        reference="https://huggingface.co/datasets/clarin-knext/arguana-pl",
        dataset={
            "path": "clarin-knext/arguana-pl",
            "revision": "63fc86750af76253e8c760fc9e534bbf24d260a2",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{wojtasik2024beirpl,
      title={BEIR-PL: Zero Shot Information Retrieval Benchmark for the Polish Language}, 
      author={Konrad Wojtasik and Vadim Shishkin and Kacper Wołowiec and Arkadiusz Janz and Maciej Piasecki},
      year={2024},
      eprint={2305.19840},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}""",
        n_samples=None,
        avg_character_length=None,
    )
