from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TopiOCQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TopiOCQA",
        dataset={
            "path": "mteb/TopiOCQA",
            "revision": "3f96cf4eb4d53dce45603e217cb050d156f5a443",
        },
        reference="https://mcgill-nlp.github.io/topiocqa",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            "is information-seeking conversational dataset with challenging topic switching phenomena. "
            "It consists of conversation histories along with manually labelled relevant/gold passage."
        ),
        type="Retrieval",
        category="t2t",
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
        bibtex_citation=r"""
@article{adlakha2022topiocqa,
  author = {Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  doi = {10.1162/tacl_a_00471},
  issn = {2307-387X},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {468–483},
  publisher = {MIT Press},
  title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
  url = {https://doi.org/10.1162/tacl_a_00471},
  volume = {10},
  year = {2022},
}
""",
    )


class TopiOCQARetrievalHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TopiOCQAHardNegatives",
        dataset={
            "path": "mteb/TopiOCQA_validation_top_250_only_w_correct-v2",
            "revision": "b4cc09fb8bb3a9e0ce0f94dc69c96397a2a47c18",
        },
        reference="https://mcgill-nlp.github.io/topiocqa",
        description=(
            "TopiOCQA (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            "is information-seeking conversational dataset with challenging topic switching phenomena. "
            "It consists of conversation histories along with manually labelled relevant/gold passage. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct."
        ),
        type="Retrieval",
        category="t2t",
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
        bibtex_citation=r"""
@article{adlakha2022topiocqa,
  author = {Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  doi = {10.1162/tacl_a_00471},
  issn = {2307-387X},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {468–483},
  publisher = {MIT Press},
  title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
  url = {https://doi.org/10.1162/tacl_a_00471},
  volume = {10},
  year = {2022},
}
""",
        adapted_from=["TopiOCQA"],
    )
