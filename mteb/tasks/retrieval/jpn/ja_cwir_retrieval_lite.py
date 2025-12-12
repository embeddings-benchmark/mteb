from datasets import Dataset, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class JaCWIRRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaCWIRRetrievalLite",
        dataset={
            "path": "sbintuitions/JMTEB-lite",
            "revision": "main",
            "trust_remote_code": True,
        },
        description=(
            "JaCWIR (Japanese Casual Web IR) is a dataset consisting of questions and webpage meta descriptions "
            "collected from Hatena Bookmark. This is the lightweight version with a reduced corpus "
            "(302,638 documents) constructed using hard negatives from 5 high-performance models."
        ),
        reference="https://huggingface.co/datasets/hotchpotch/JaCWIR",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-01-01"),
        domains=["Web", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
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

@misc{yuichi-tateno-2024-jacwir,
  author = {Yuichi Tateno},
  title = {JaCWIR: Japanese Casual Web IR - 日本語情報検索評価のための小規模でカジュアルなWebタイトルと概要のデータセット},
  url = {https://huggingface.co/datasets/hotchpotch/JaCWIR},
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
                "jacwir-retrieval-query",
                split=split,
                trust_remote_code=True,
            )
            corpus = load_dataset(
                self.metadata.dataset["path"],
                "jacwir-retrieval-corpus",
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
                top_ranked=None,  # Retrieval task searches entire corpus
            )

        self.data_loaded = True
