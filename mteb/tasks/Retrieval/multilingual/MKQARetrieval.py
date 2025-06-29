from __future__ import annotations

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = [
    "ar",
    "da",
    "de",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hu",
    "it",
    "ja",
    "ko",
    "km",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tr",
    "vi",
    "zh_cn",
    "zh_hk",
    "zh_tw",
]

_LANGUAGE_MAPPING = {
    "ar": "ara-Arab",
    "da": "dan-Latn",
    "de": "deu-Latn",
    "en": "eng-Latn",
    "es": "spa-Latn",
    "fi": "fin-Latn",
    "fr": "fra-Latn",
    "he": "heb-Hebr",
    "hu": "hun-Latn",
    "it": "ita-Latn",
    "ja": "jpn-Jpan",
    "ko": "kor-Kore",
    "km": "khm-Khmr",
    "ms": "msa-Latn",
    "nl": "nld-Latn",
    "no": "nor-Latn",
    "pl": "pol-Latn",
    "pt": "por-Latn",
    "ru": "rus-Cyrl",
    "sv": "swe-Latn",
    "th": "tha-Thai",
    "tr": "tur-Latn",
    "vi": "vie-Latn",
    "zh_cn": "zho-Hans",
    "zh_hk": "zho-Hant",
    "zh_tw": "zho-Hant",
}


_EVAL_LANGS = {
    f"{s_lang}-{t_lang}": [_LANGUAGE_MAPPING[s_lang], _LANGUAGE_MAPPING[t_lang]]
    for s_lang in _LANGUAGES
    for t_lang in _LANGUAGES
}


class MKQARetrieval(AbsTaskRetrieval, MultilingualTask):
    metadata = TaskMetadata(
        name="MKQARetrieval",
        description="""Multilingual Knowledge Questions & Answers (MKQA)contains 10,000 queries sampled from the Google Natural Questions dataset.
        For each query we collect new passage-independent answers. These queries and answers are then human translated into 25 Non-English languages.""",
        reference="https://github.com/apple/ml-mkqa",
        dataset={
            "path": "apple/mkqa",
            "revision": "325131889721ae0ed885b76ecb8011369d75abad",
            "trust_remote_code": True,
            "name": "mkqa",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        date=("2020-01-01", "2020-12-31"),
        eval_splits=["train"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        domains=["Written"],
        task_subtypes=["Question answering"],
        license="cc-by-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{mkqa,
  author = {Shayne Longpre and Yi Lu and Joachim Daiber},
  title = {MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering},
  url = {https://arxiv.org/pdf/2007.15207.pdf},
  year = {2020},
}
        """,
    )

    def load_data(self, **kwargs):
        """In this retrieval datasets, corpus is in lang XX and queries in lang YY."""
        if self.data_loaded:
            return

        self.queries, self.corpus, self.relevant_docs = {}, {}, {}

        ds = datasets.load_dataset(
            **self.metadata_dict["dataset"],
        )

        for lang_pair in self.hf_subsets:
            source_lang_abb, target_lang_abb = lang_pair.split("-")

            self.queries[lang_pair] = {}
            self.corpus[lang_pair] = {}
            self.relevant_docs[lang_pair] = {}

            for eval_split in self.metadata.eval_splits:
                self.queries[lang_pair][eval_split] = {}
                self.corpus[lang_pair][eval_split] = {}
                self.relevant_docs[lang_pair][eval_split] = {}

                split_data = ds[eval_split]

                query_ids = {
                    query: f"Q{i}"
                    for i, query in enumerate(
                        {entry[source_lang_abb] for entry in split_data["queries"]}
                    )
                }

                context_texts = {
                    hit["text"]
                    for entry in split_data["answers"]
                    for hit in entry[target_lang_abb]
                }

                context_ids = {text: f"C{i}" for i, text in enumerate(context_texts)}

                for row in split_data:
                    query = row["queries"][source_lang_abb]
                    contexts = [
                        entry["text"] for entry in row["answers"][target_lang_abb]
                    ]

                    if query is None or None in contexts:
                        continue

                    query_id = query_ids[query]
                    for context in contexts:
                        context_id = context_ids[context]
                        self.queries[lang_pair][eval_split][query_id] = query
                        self.corpus[lang_pair][eval_split][context_id] = {
                            "title": "",
                            "text": context,
                        }
                        if query_id not in self.relevant_docs[lang_pair][eval_split]:
                            self.relevant_docs[lang_pair][eval_split][query_id] = {}
                        self.relevant_docs[lang_pair][eval_split][query_id][
                            context_id
                        ] = 1

            self.data_loaded = True
