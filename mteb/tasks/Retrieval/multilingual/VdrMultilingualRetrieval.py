from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "de": ["deu-Latn"],
    "it": ["ita-Latn"],
}
_EVAL_SPLIT = "train"  # It is test split only, but given name as train.


def _load_vdr_multilingual_data(
    path: str,
    langs: list,
    split: str,
    cache_dir: str = None,
    revision: str = None,
    trust_remote_code: bool = False,
):
    corpus = {lang_code: {split: {}} for lang_code in langs}
    queries = {lang_code: {split: {}} for lang_code in langs}
    relevant_docs = {lang_code: {split: {}} for lang_code in langs}

    for lang_code in langs:
        dataset = datasets.load_dataset(
            path=path,
            name=lang_code,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        if split not in dataset:
            continue

        for idx, record in enumerate(dataset[split]):
            doc_id = f"doc-{record['id']}"
            corpus[lang_code][split][doc_id] = {
                "text": record.get("query", ""),
                "image": record.get("image", None),
                "modality": "text,image" if record.get("image") is not None else "text",
            }

            query_id = f"query-{record['id']}"
            queries[lang_code][split][query_id] = {
                "text": record.get("query", ""),
                "image": record.get("image", None),
                "modality": "text,image" if record.get("image") is not None else "text",
            }

            if query_id not in relevant_docs[lang_code][split]:
                relevant_docs[lang_code][split][query_id] = {}
            relevant_docs[lang_code][split][query_id][doc_id] = 1

    corpus = datasets.DatasetDict(corpus)
    queries = datasets.DatasetDict(queries)
    relevant_docs = datasets.DatasetDict(relevant_docs)

    return corpus, queries, relevant_docs


class VDRMultilingualRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VDRMultilingualRetrieval",
        description="Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German",
        reference="https://huggingface.co/datasets/llamaindex/vdr-multilingual-test",
        dataset={
            "path": "llamaindex/vdr-multilingual-test",
            "revision": "9e26ae152f5950ab1a5ff1c58edade3acc894793"
        },
        type="Retrieval",
        category="it2it",
        modalities=["text", "image"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=(
            "2025-10-01",
            "2025-08-01",
        ),  # Not Specified excatly in the dataset and blog
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@misc{llamaindex2024vdrmultilingual,
      title={Visual Document Retrieval Goes Multilingual},
      author={LlamaIndex},
      year={2025},
      howpublished={https://huggingface.co/datasets/llamaindex/vdr-multilingual-test},
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = _load_vdr_multilingual_data(
            path=self.metadata_dict["dataset"]["path"],
            langs=self.hf_subsets,
            split=self.metadata_dict["eval_splits"][0],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"].get("revision", None),
            trust_remote_code=self.metadata_dict["dataset"].get(
                "trust_remote_code", False
            ),
        )

        self.data_loaded = True
