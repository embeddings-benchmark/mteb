from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MrTyDiJaRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MrTyDiJaRetrievalLite",
        dataset={
            "path": "sbintuitions/JMTEB-lite",
            "revision": "main",
            "trust_remote_code": True,
        },
        description=(
            "Mr.TyDi is a multilingual benchmark dataset built on TyDi for document retrieval tasks. "
            "This is the lightweight Japanese version with a reduced corpus (93,382 documents) constructed using "
            "hard negatives from 5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/castorini/mr-tydi",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jmteb_lite,
  author = {Li, Shengzhe and Ohagi, Masaya and Ri, Ryokan and Fukuchi, Akihiko and Shibata, Tomohide
and Kawahara, Daisuke},
  howpublished = {\url{https://huggingface.co/datasets/sbintuitions/JMTEB-lite}},
  title = {{J}{M}{T}{E}{B}-lite: {T}he {L}ightweight {V}ersion of {JMTEB}},
  year = {2025},
}

@article{mrtydi,
  author = {Xinyu Zhang and Xueguang Ma and Peng Shi and Jimmy Lin},
  journal = {arXiv:2108.08787},
  title = {{Mr. TyDi}: A Multi-lingual Benchmark for Dense Retrieval},
  year = {2021},
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
                "mrtydi-query",
                split=split,
                trust_remote_code=True,
            )
            corpus = load_dataset(
                self.metadata.dataset["path"],
                "mrtydi-corpus",
                split="corpus",
                trust_remote_code=True,
            )

            relevant_docs = {}

            queries_list = []
            for query in queries:
                qid = str(query["qid"])
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
                top_ranked=None,  # Retrieval task searches entire corpus
            )

        self.data_loaded = True
