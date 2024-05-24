from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class RuMMarcoRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="RuMMarcoRetrieval",
        dataset={
            "path": "ai-forever/ru-mmarco-retrieval",
            "revision": "18d1c2b1ab2a7e8920614329e19ab4c513113d7e",
        },
        description="mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset",
        reference="https://arxiv.org/abs/2108.13897",
        type="Retrieval",
        category="s2p",
        eval_splits=["dev"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2019-01-01"),
        form=["written"],
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        socioeconomic_status="mixed",
        annotations_creators="derived",
        dialect=[],
        text_creation="machine-translated",
        bibtex_citation="""@misc{bonifacio2022mmarco,
        title={mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset}, 
        author={Luiz Bonifacio 
                    and Vitor Jeronymo 
                    and Hugo Queiroz Abonizio 
                    and Israel Campiotti 
                    and Marzieh Fadaee 
                    and Roberto Lotufo 
                    and Rodrigo Nogueira},
        year={2022},
        eprint={2108.13897},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
        }""",
        n_samples={"dev": 7437},
        avg_character_length={"dev": 385.9},
    )
