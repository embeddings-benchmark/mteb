import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class DanFeverRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="DanFeverRetrieval",
        dataset={
            "path": "strombergnlp/danfever",
            "revision": "3b17b754ed5bf356582b93ec11d1c72469f7bb9c",
        },
        description="A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset. DanFeverRetrieval fixed an issue in DanFever where some corpus entries were incorrectly removed.",
        reference="https://aclanthology.org/2021.nodalida-main.47/",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="CC BY-SA 4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{norregaard-derczynski-2021-danfever,
    title = "{D}an{FEVER}: claim verification dataset for {D}anish",
    author = "N{\o}rregaard, Jeppe  and
      Derczynski, Leon",
    editor = "Dobnik, Simon  and
      {\O}vrelid, Lilja",
    booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may # " 31--2 " # jun,
    year = "2021",
    address = "Reykjavik, Iceland (Online)",
    publisher = {Link{\"o}ping University Electronic Press, Sweden},
    url = "https://aclanthology.org/2021.nodalida-main.47",
    pages = "422--428",
    abstract = "We present a dataset, DanFEVER, intended for multilingual misinformation research. The dataset is in Danish and has the same format as the well-known English FEVER dataset. It can be used for testing methods in multilingual settings, as well as for creating models in production for the Danish language.",
}
""",
        descriptive_stats={
            "n_samples": {"train": 8897},
            "avg_character_length": {
                "train": {
                    "average_document_length": 312.1117274167987,
                    "average_query_length": 50.26957476855484,
                    "num_documents": 2524,
                    "num_queries": 6373,
                    "average_relevant_docs_per_query": 0.48721167425074535,
                }
            },
        },
        task_subtypes=["Claim verification"],
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document data like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            self.corpus[split] = {}
            self.relevant_docs[split] = {}
            self.queries[split] = {}

            ds = self.dataset[split]
            claims = ds["claim"]
            evidences = ds["evidence_extract"]
            labels = ds["label"]
            class_labels = ds.features["label"].names

            for claim, evidence, label_id in zip(claims, evidences, labels):
                claim_is_supported = class_labels[label_id] == "Supported"

                sim = (
                    1 if claim_is_supported else 0
                )  # negative for refutes claims - is that what we want?

                if claim not in text2id:
                    text2id[claim] = str(len(text2id))
                if evidence not in text2id:
                    text2id[evidence] = len(text2id)

                claim_id = "Q" + str(text2id[claim])
                evidence_id = "C" + str(text2id[evidence])

                self.queries[split][claim_id] = claim
                self.corpus[split][evidence_id] = {"title": "", "text": evidence}

                if claim_id not in self.relevant_docs[split]:
                    self.relevant_docs[split][claim_id] = {}

                self.relevant_docs[split][claim_id][evidence_id] = sim


class DanFever(AbsTaskRetrieval):
    ignore_identical_ids = True
    superseded_by = "DanFeverRetrieval"

    metadata = TaskMetadata(
        name="DanFEVER",
        dataset={
            "path": "strombergnlp/danfever",
            "revision": "5d01e3f6a661d48e127ab5d7e3aaa0dc8331438a",
            "trust_remote_code": True,
        },
        description="A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.",
        reference="https://aclanthology.org/2021.nodalida-main.47/",
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="CC BY-SA 4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
@inproceedings{norregaard-derczynski-2021-danfever,
    title = "{D}an{FEVER}: claim verification dataset for {D}anish",
    author = "N{\o}rregaard, Jeppe  and
      Derczynski, Leon",
    editor = "Dobnik, Simon  and
      {\O}vrelid, Lilja",
    booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may # " 31--2 " # jun,
    year = "2021",
    address = "Reykjavik, Iceland (Online)",
    publisher = {Link{\"o}ping University Electronic Press, Sweden},
    url = "https://aclanthology.org/2021.nodalida-main.47",
    pages = "422--428",
    abstract = "We present a dataset, DanFEVER, intended for multilingual misinformation research. The dataset is in Danish and has the same format as the well-known English FEVER dataset. It can be used for testing methods in multilingual settings, as well as for creating models in production for the Danish language.",
}
""",
        descriptive_stats={
            "n_samples": {"train": 8897},
            "avg_character_length": {
                "train": {
                    "average_document_length": 312.1117274167987,
                    "average_query_length": 50.26957476855484,
                    "num_documents": 2524,
                    "num_queries": 6373,
                    "average_relevant_docs_per_query": 0.48721167425074535,
                }
            },
        },
        task_subtypes=["Claim verification"],
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document data like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            self.corpus[split] = {}
            self.relevant_docs[split] = {}
            self.queries[split] = {}

            ds = self.dataset[split]
            claims = ds["claim"]
            evidences = ds["evidence_extract"]
            labels = ds["label"]
            class_labels = ds.features["label"].names

            for claim, evidence, label_id in zip(claims, evidences, labels):
                claim_is_supported = class_labels[label_id] == "Supported"

                sim = (
                    1 if claim_is_supported else 0
                )  # negative for refutes claims - is that what we want?

                if claim not in text2id:
                    text2id[claim] = str(len(text2id))
                if evidence not in text2id:
                    text2id[evidence] = len(text2id)

                claim_id = str(text2id[claim])
                evidence_id = str(text2id[evidence])

                self.queries[split][claim_id] = claim
                self.corpus[split][evidence_id] = {"title": "", "text": evidence}

                if claim_id not in self.relevant_docs[split]:
                    self.relevant_docs[split][claim_id] = {}

                self.relevant_docs[split][claim_id][evidence_id] = sim
