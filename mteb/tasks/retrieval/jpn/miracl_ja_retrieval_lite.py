from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MIRACLJaRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLJaRetrievalLite",
        dataset={
            "path": "sbintuitions/JMTEB-lite",
            "revision": "main",
            "trust_remote_code": True,
        },
        description=(
            "MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual "
            "retrieval dataset. This is the lightweight Japanese version with a reduced corpus (105,064 documents) "
            "constructed using hard negatives from 5 high-performance models."
        ),
        reference="https://project-miracl.github.io/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2022-06-01", "2025-01-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{10.1162/tacl_a_00595,
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David
and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  doi = {10.1162/tacl_a_00595},
  journal = {Transactions of the Association for Computational Linguistics},
  pages = {1114-1131},
  title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
  volume = {11},
  year = {2023},
}

@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
and Kawahara, Daisuke},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  year = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        """Load JMTEB-lite dataset."""
        if self.data_loaded:
            return

        for split in self.metadata.eval_splits:
            queries = load_dataset(
                self.metadata.dataset["path"],
                "miracl-retrieval-query",
                split=split,
                trust_remote_code=True,
            )
            corpus = load_dataset(
                self.metadata.dataset["path"],
                "miracl-retrieval-corpus",
                split="corpus",
                trust_remote_code=True,
            )

            relevant_docs = {}

            queries_list = []
            for i, query in enumerate(queries):
                qid = split + "_" + str(i + 1)
                queries_list.append({"id": qid, "text": query["query"]})

                if "relevant_docs" in query:
                    relevant_docs[qid] = {
                        str(doc_id): 1 for doc_id in query["relevant_docs"]
                    }

            corpus_list = []
            for doc in corpus:
                corpus_list.append(
                    {
                        "id": str(doc["docid"]),
                        "text": doc["text"],
                        "title": doc.get("title", ""),
                    }
                )

            self.dataset["default"][split] = RetrievalSplitData(
                corpus=Dataset.from_list(corpus_list),
                queries=Dataset.from_list(queries_list),
                relevant_docs=relevant_docs,
                top_ranked=None,  # Retrieval task searches entire corpus, not pre-filtered
            )

        self.data_loaded = True
