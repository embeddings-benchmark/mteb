from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class GermanDPR(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    _LANGUAGE = "de"

    metadata = TaskMetadata(
        name="GermanDPR",
        description="GermanDPR is a German Question Answering dataset for open-domain QA. It associates questions with a textual context containing the answer",
        reference="https://www.deepset.ai/germanquad",
        hf_hub_name="deepset/germanquad",
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=[_LANGUAGE],
        main_score="ndcg_at_10",
        revision="5129d02422a66be600ac89cd3e8531b4f97d347d",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )

    @staticmethod
    def _format_documents(docs, id_prefix="", existing_docs=None):
        if existing_docs is None:
            existing_docs = dict()
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
            self.metadata_dict["hf_hub_name"],
            revision=self.metadata_dict.get("revision", None),
            split=self._EVAL_SPLIT,
        )
        corpus = dict()
        queries = dict()
        relevant_docs = dict()
        all_docs = dict()
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
        self.queries = {self._EVAL_SPLIT: queries}
        self.corpus = {self._EVAL_SPLIT: corpus}
        self.relevant_docs = {self._EVAL_SPLIT: relevant_docs}

        self.data_loaded = True
