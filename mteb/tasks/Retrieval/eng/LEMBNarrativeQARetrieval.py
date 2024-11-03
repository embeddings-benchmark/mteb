from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class LEMBNarrativeQARetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBNarrativeQARetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "narrativeqa",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("narrativeqa subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("1000-01-01", "2017-12-31"),
        domains=["Fiction", "Non-fiction", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
            @article{kocisky-etal-2018-narrativeqa,
            title = "The {N}arrative{QA} Reading Comprehension Challenge",
            author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
            Schwarz, Jonathan  and
            Blunsom, Phil  and
            Dyer, Chris  and
            Hermann, Karl Moritz  and
            Melis, G{\'a}bor  and
            Grefenstette, Edward",
            editor = "Lee, Lillian  and
            Johnson, Mark  and
            Toutanova, Kristina  and
            Roark, Brian",
            journal = "Transactions of the Association for Computational Linguistics",
            volume = "6",
            year = "2018",
            address = "Cambridge, MA",
            publisher = "MIT Press",
            url = "https://aclanthology.org/Q18-1023",
            doi = "10.1162/tacl_a_00023",
            pages = "317--328",
            abstract = "",
        }
        """,
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "queries"
        ]  # dict_keys(['qid', 'text'])
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "corpus"
        ]  # dict_keys(['doc_id', 'text'])
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata_dict["dataset"])[
            "qrels"
        ]  # dict_keys(['qid', 'doc_id'])
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}

        self.data_loaded = True
