from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class SCIDOCSPL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="SCIDOCS-PL",
        description="SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.",
        reference="https://allenai.org/data/scidocs",
        dataset={
            "path": "clarin-knext/scidocs-pl",
            "revision": "45452b03f05560207ef19149545f168e596c9337",
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
