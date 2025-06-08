from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GermanDPR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="GermanDPR",
        description="GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer",
        reference="https://huggingface.co/datasets/deepset/germandpr",
        dataset={
            "path": "deepset/germandpr",
            "revision": "5129d02422a66be600ac89cd3e8531b4f97d347d",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["deu-Latn"],
        main_score="ndcg_at_10",
        date=("2020-05-19", "2021-04-26"),
        domains=["Written", "Non-fiction", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{möller2021germanquad,
  archiveprefix = {arXiv},
  author = {Timo Möller and Julian Risch and Malte Pietsch},
  eprint = {2104.12741},
  primaryclass = {cs.CL},
  title = {GermanQuAD and GermanDPR: Improving Non-English Question Answering and Passage Retrieval},
  year = {2021},
}
""",
    )

    @staticmethod
    def _format_documents(docs, id_prefix="", existing_docs=None):
        if existing_docs is None:
            existing_docs = {}
        result = {}
        for i, (title, content) in enumerate(zip(docs["title"], docs["text"])):
            formatted_content = content.split("==\n")[-1].replace("\n", " ").lstrip()
            if formatted_content in existing_docs:
                id_value = existing_docs[formatted_content]
            else:
                id_value = f"{id_prefix}{i}"
                existing_docs[formatted_content] = id_value
            result[id_value] = {"title": title, "text": formatted_content}
        return result

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = datasets.load_dataset(
            split=self._EVAL_SPLIT,
            **self.metadata_dict["dataset"],
        )
        corpus = {}
        queries = {}
        relevant_docs = {}
        all_docs = {}
        for i, row in enumerate(data):
            q_id = f"q_{i}"
            queries[q_id] = row["question"]
            pos_docs = self._format_documents(
                row["positive_ctxs"], id_prefix=f"doc_{i}_p_", existing_docs=all_docs
            )
            corpus.update(pos_docs)
            neg_docs = self._format_documents(
                row["hard_negative_ctxs"],
                id_prefix=f"doc_{i}_n_",
                existing_docs=all_docs,
            )
            corpus.update(neg_docs)
            relevant_docs[q_id] = {k: 1 for k in pos_docs}
        corpus = {
            key: doc.get("title", "") + " " + doc["text"] for key, doc in corpus.items()
        }
        self.queries = {self._EVAL_SPLIT: queries}
        self.corpus = {self._EVAL_SPLIT: corpus}
        self.relevant_docs = {self._EVAL_SPLIT: relevant_docs}

        self.data_loaded = True
