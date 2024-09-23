from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class TopiOCQARetrievalFast(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TopiOCQA-Fast",
        dataset={
            "path": f"mteb/TopiOCQA_validation_top_250_only_w_correct",
            "revision": "latest",
            "trust_remote_code": True,
        },
        reference="https://mcgill-nlp.github.io/topiocqa",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            + "is information-seeking conversational dataset with challenging topic switching phenomena. "
            + "It consists of conversation histories along with manually labelled relevant/gold passage."
        ),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2021-03-01", "2021-07-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @misc{adlakha2022topiocqa,
      title={TopiOCQA: Open-domain Conversational Question Answering with Topic Switching}, 
      author={Vaibhav Adlakha and Shehzaad Dhuliawala and Kaheer Suleman and Harm de Vries and Siva Reddy},
      year={2022},
      eprint={2110.00768},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
        }
        """,
        descriptive_stats={
            "n_samples": None,
            "avg_character_length": {
                "validation": {"average_document_length": 525.5101748190006, "average_query_length": 12.85, "num_documents": 141575, "num_queries": 1000, "average_relevant_docs_per_query": 1.0}
            },
        },
    )
