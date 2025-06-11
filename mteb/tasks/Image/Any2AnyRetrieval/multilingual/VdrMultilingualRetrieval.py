from __future__ import annotations

import datasets
from datasets import Dataset, DatasetDict

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGS = {
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "de": ["deu-Latn"],
    "it": ["ita-Latn"],
}
_EVAL_SPLIT = "train"  # It is test split only, but given name as train on HF


def _load_vdr_multilingual_data(
    path: str,
    langs: list,
    split: str,
    cache_dir: str = None,
    revision: str = None,
    trust_remote_code: bool = False,
):
    """Load data from the VDR Multilingual dataset."""
    corpus_dict = {}
    queries_dict = {}
    relevant_docs_dict = {}

    for lang_code in langs:
        dataset = datasets.load_dataset(
            path=path,
            name=lang_code,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        corpus_records = []
        queries_records = []
        relevant_dict = {}

        for idx, record in enumerate(dataset):
            doc_id = f"doc-{record['id']}"
            query_id = f"query-{record['id']}"
            has_query = record.get("query") is not None

            corpus_records.append(
                {
                    "id": doc_id,
                    "image": record.get("image"),
                    "modality": "image",
                }
            )

            if has_query:
                queries_records.append(
                    {
                        "id": query_id,
                        "text": record.get("query", ""),
                        "modality": "text",
                    }
                )

                if query_id not in relevant_dict:
                    relevant_dict[query_id] = {}
                relevant_dict[query_id][doc_id] = 1

        if lang_code not in corpus_dict:
            corpus_dict[lang_code] = {}
        if lang_code not in queries_dict:
            queries_dict[lang_code] = {}
        if lang_code not in relevant_docs_dict:
            relevant_docs_dict[lang_code] = {}

        corpus_dict[lang_code][split] = Dataset.from_list(corpus_records)
        queries_dict[lang_code][split] = Dataset.from_list(queries_records)
        relevant_docs_dict[lang_code][split] = relevant_dict

    corpus_dataset_dict = {}
    queries_dataset_dict = {}
    relevant_docs_dataset_dict = {}

    for lang_code in langs:
        if (
            lang_code in corpus_dict
            and lang_code in queries_dict
            and lang_code in relevant_docs_dict
        ):
            corpus_dataset_dict[lang_code] = DatasetDict(corpus_dict[lang_code])
            queries_dataset_dict[lang_code] = DatasetDict(queries_dict[lang_code])
            relevant_docs_dataset_dict[lang_code] = relevant_docs_dict[lang_code]

    return corpus_dataset_dict, queries_dataset_dict, relevant_docs_dataset_dict


class VDRMultilingualRetrieval(MultilingualTask, AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="VDRMultilingualRetrieval",
        description="Multilingual Visual Document retrieval Dataset covering 5 languages: Italian, Spanish, English, French and German",
        reference="https://huggingface.co/datasets/llamaindex/vdr-multilingual-test",
        dataset={
            "path": "llamaindex/vdr-multilingual-test",
            "revision": "9e26ae152f5950ab1a5ff1c58edade3acc894793",
        },
        type="Retrieval",
        category="it2it",
        modalities=["text", "image"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=_LANGS,
        main_score="ndcg_at_5",
        date=(
            "2025-01-01",
            "2025-01-10",
        ),  # Not Specified exactly in the dataset and blog
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{llamaindex2024vdrmultilingual,
  author = {LlamaIndex},
  howpublished = {https://huggingface.co/datasets/llamaindex/vdr-multilingual-test},
  title = {Visual Document Retrieval Goes Multilingual},
  year = {2025},
}
""",
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
