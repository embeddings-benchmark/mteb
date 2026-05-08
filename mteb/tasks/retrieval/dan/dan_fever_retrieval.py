import datasets
from datasets import Dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


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
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{norregaard-derczynski-2021-danfever,
  address = {Reykjavik, Iceland (Online)},
  author = {N{\o}rregaard, Jeppe  and
Derczynski, Leon},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {422--428},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {{D}an{FEVER}: claim verification dataset for {D}anish},
  url = {https://aclanthology.org/2021.nodalida-main.47},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim in Danish, retrieve documents that support the claim"
        },
        task_subtypes=["Claim verification"],
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        hf_dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform(hf_dataset)
        self.data_loaded = True

    def dataset_transform(
        self, hf_dataset, num_proc: int | None = None, **kwargs
    ) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.dataset = {subset: {split: {"corpus": Dataset, "queries": Dataset, "relevant_docs": dict, "top_ranked": None}}}
        """
        self.dataset = {}
        text2id = {}

        for split in hf_dataset:
            corpus_dict = {}
            relevant_docs = {}
            queries_dict = {}

            ds = hf_dataset[split]
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

                queries_dict[claim_id] = claim
                corpus_dict[evidence_id] = {"title": "", "text": evidence}

                if claim_id not in relevant_docs:
                    relevant_docs[claim_id] = {}

                relevant_docs[claim_id][evidence_id] = sim

            corpus_dataset = Dataset.from_list(
                [
                    {"id": k, "text": v["text"], "title": v["title"]}
                    for k, v in corpus_dict.items()
                ]
            )
            queries_dataset = Dataset.from_list(
                [{"id": k, "text": v} for k, v in queries_dict.items()]
            )

            self.dataset.setdefault("default", {})[split] = {
                "corpus": corpus_dataset,
                "queries": queries_dataset,
                "relevant_docs": relevant_docs,
                "top_ranked": None,
            }


class DanFever(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="DanFEVER",
        dataset={
            "path": "mteb/DanFEVER",
            "revision": "5222ad7225b04f192be495bbd83b198569eef810",
        },
        description="A Danish dataset intended for misinformation research. It follows the same format as the English FEVER dataset.",
        reference="https://aclanthology.org/2021.nodalida-main.47/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2021-12-31"),  # best guess
        domains=["Encyclopaedic", "Non-fiction", "Spoken"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{norregaard-derczynski-2021-danfever,
  address = {Reykjavik, Iceland (Online)},
  author = {N{\o}rregaard, Jeppe  and
Derczynski, Leon},
  booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)},
  editor = {Dobnik, Simon  and
{\O}vrelid, Lilja},
  month = may # { 31--2 } # jun,
  pages = {422--428},
  publisher = {Link{\"o}ping University Electronic Press, Sweden},
  title = {{D}an{FEVER}: claim verification dataset for {D}anish},
  url = {https://aclanthology.org/2021.nodalida-main.47},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim in Danish, retrieve documents that support the claim"
        },
        task_subtypes=["Claim verification"],
        superseded_by="DanFeverRetrieval",
    )
