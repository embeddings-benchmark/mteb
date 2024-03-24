from __future__ import annotations

import datasets

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import MultilingualTask
from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = "test"
_LANGS = ["de", "es"]


def _load_miracl_data(
    path: str, langs: list, split: str, cache_dir: str = None, revision: str = None
):
    queries = {lang: {split: {}} for lang in langs}
    corpus = {lang: {split: {}} for lang in langs}
    relevant_docs = {lang: {split: {}} for lang in langs}

    for lang in langs:
        data = datasets.load_dataset(
            path,
            lang,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        # Generate unique IDs for queries and documents
        query_id_counter = 1
        document_id_counter = 1

        for row in data:
            query_text = row["query"]
            positive_texts = row["positive"]
            negative_texts = row["negative"]

            # Assign unique ID to the query
            query_id = f"Q{query_id_counter}"
            queries[lang][split][query_id] = query_text
            query_id_counter += 1

            # Add positive and negative texts to corpus with unique IDs
            for text in positive_texts + negative_texts:
                doc_id = f"D{document_id_counter}"
                corpus[lang][split][doc_id] = {"text": text}
                document_id_counter += 1

                # Add relevant document information to relevant_docs for positive texts only
                if text in positive_texts:
                    if query_id not in relevant_docs[lang][split]:
                        relevant_docs[lang][split][query_id] = {}
                    relevant_docs[lang][split][query_id][doc_id] = 1

        corpus = datasets.DatasetDict(corpus)
        queries = datasets.DatasetDict(queries)
        relevant_docs = datasets.DatasetDict(relevant_docs)

        return corpus, queries, relevant_docs


class MIRACLRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MIRACLRetrieval",
        description="MIRACLRetrieval",
        reference=None,
        hf_hub_name="jinaai/miracl",
        type="Retrieval",
        category="s2p",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_10",
        revision="d28a029f35c4ff7f616df47b0edf54e6882395e6",
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

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_miracl_data(
            path=self.metadata_dict["hf_hub_name"],
            langs=self.langs,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["revision"],
        )

        self.data_loaded = True
