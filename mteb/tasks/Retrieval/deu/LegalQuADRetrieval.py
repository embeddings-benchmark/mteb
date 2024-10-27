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
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
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
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "test": {
                    "average_document_length": 19481.955,
                    "average_query_length": 71.965,
                    "num_documents": 200,
                    "num_queries": 200,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
