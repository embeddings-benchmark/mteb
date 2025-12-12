from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class JaqketRetrievalLite(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="JaqketRetrievalLite",
        dataset={
            "path": "sbintuitions/JMTEB-lite",
            "revision": "main",
            "trust_remote_code": True,
        },
        description=(
            "JAQKET (JApanese Questions on Knowledge of EnTities) is a QA dataset created based on quiz questions. "
            "This is the lightweight version with a reduced corpus (65,802 documents) constructed using "
            "hard negatives from 5 high-performance models."
        ),
        reference="https://github.com/kumapo/JAQKET-dataset",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="ndcg_at_10",
        date=("2023-10-09", "2025-01-01"),
        domains=["Encyclopaedic", "Non-fiction", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
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

@inproceedings{Kurihara_nlp2020,
  author = {鈴木正敏 and 鈴木潤 and 松田耕史 and ⻄田京介 and 井之上直也},
  booktitle = {言語処理学会第26回年次大会},
  note = {in Japanese},
  title = {JAQKET: クイズを題材にした日本語 QA データセットの構築},
  url = {https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf},
  year = {2020},
}
""",
    )

    def load_data(self, **kwargs):
        """Load JMTEB-lite dataset."""
        if self.data_loaded:
            return

        # JMTEB-lite uses 'jaqket-query' and 'jaqket-corpus' configs
        for split in self.metadata.eval_splits:
            queries = load_dataset(
                self.metadata.dataset["path"],
                "jaqket-query",
                split=split,
                trust_remote_code=True,
            )
            corpus = load_dataset(
                self.metadata.dataset["path"],
                "jaqket-corpus",
                split="corpus",
                trust_remote_code=True,
            )

            # Build relevant_docs from queries dataset
            relevant_docs = {}

            queries_list = []
            for query in queries:
                qid = str(query["qid"])
                queries_list.append({"id": qid, "text": query["query"]})

                # Build relevant docs (relevance score = 1 for all relevant docs)
                if "relevant_docs" in query:
                    relevant_docs[qid] = {
                        str(doc_id): 1 for doc_id in query["relevant_docs"]
                    }

            # Prepare corpus in MTEB format
            corpus_list = []
            for doc in corpus:
                corpus_list.append(
                    {
                        "id": str(doc["docid"]),
                        "text": doc["text"],
                        "title": doc.get("title", ""),
                    }
                )

            from datasets import Dataset

            self.dataset["default"][split] = RetrievalSplitData(
                corpus=Dataset.from_list(corpus_list),
                queries=Dataset.from_list(queries_list),
                relevant_docs=relevant_docs,
                top_ranked=None,  # Retrieval task searches entire corpus
            )

        self.data_loaded = True
