from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LegalQuAD(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalQuAD",
        description="The dataset consists of questions and legal documents in German.",
        reference="https://github.com/Christoph911/AIKE2021_Appendix",
        dataset={
            "path": "mteb/LegalQuAD",
            "revision": "37aa6cfb01d48960b0f8e3f17d6e3d99bf1ebc3e",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Question answering"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation="""@INPROCEEDINGS{9723721,
  author={Hoppe, Christoph and Pelkmann, David and Migenda, Nico and Hötte, Daniel and Schenck, Wolfram},
  booktitle={2021 IEEE Fourth International Conference on Artificial Intelligence and Knowledge Engineering (AIKE)}, 
  title={Towards Intelligent Legal Advisors for Document Retrieval and Question-Answering in German Legal Documents}, 
  year={2021},
  volume={},
  number={},
  pages={29-32},
  keywords={Knowledge engineering;Law;Semantic search;Conferences;Bit error rate;NLP;knowledge extraction;question-answering;semantic search;document retrieval;German language},
  doi={10.1109/AIKE52691.2021.00011}
  }""",
        n_samples=None,
        avg_character_length=None,
    )
