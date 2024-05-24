from typing import Dict, List

import datasets

from mteb.abstasks import AbsTaskRetrieval, CrosslingualTask, TaskMetadata

_LANGUAGES = {
    "wmt19.de.fr": ["deu-Latn", "fra-Latn"],
    "wmt19.fr.de": ["fra-Latn", "deu-Latn"],
}


def _build_lang_pair(langs: List[str]) -> str:
    """Builds a language pair separated by a dash.
    e.g., ['eng-Latn', 'deu-Latn'] -> 'eng-deu'.
    """
    return langs[0].split("-")[0] + "-" + langs[1].split("-")[0]


def extend_lang_pairs() -> Dict[str, List[str]]:
    eval_langs = {}
    for langs in _LANGUAGES.values():
        lang_pair = _build_lang_pair(langs)
        eval_langs[lang_pair] = langs
    return eval_langs


_EVAL_LANGS = extend_lang_pairs()


class CrossLingualSemanticDiscriminationWMT19(AbsTaskRetrieval, CrosslingualTask):
    metadata = TaskMetadata(
        name="CrossLingualSemanticDiscriminationWMT19",
        dataset={
            "path": "Andrianos/clsd_wmt19_21",
            "revision": "9627fbdb39b827ee5c066011ebe1e947cdb137bd",
        },
        description="Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT19 DE-FR test set",
        reference="https://huggingface.co/datasets/Andrianos/clsd_wmt19_21",
        type="Retrieval",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="recall_at_1",
        date=("2018-01-01", "2023-12-12"),
        form=["written"],
        domains=["News"],
        task_subtypes=["Cross-Lingual Semantic Discrimination"],
        license="CC BY-SA 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="LM-generated and verified",
        bibtex_citation="preprint_coming",
        n_samples={"test": 2946},
        avg_character_length={"test": 161},
    )

    def __init__(self, **kwargs):
        self.num_of_distractors = 4
        super().__init__(**kwargs)

    def load_data(self, **kwargs):
        """Generic data loader function for original clsd datasets with the format shown in "hf_dataset_link".
        Loading the hf dataset, it populates the following three variables to be used for retrieval evaluation.

        self.corpus

        self.queries

        self.relevant_docs

        Sets self.data_loaded to True.
        """
        if self.data_loaded:
            return
        queries, corpus, relevant_docs = {}, {}, {}
        dataset_raw = {}
        for split in self.metadata.eval_splits:
            for hf_subset, langs in _LANGUAGES.items():
                lang_pair = _build_lang_pair(langs)
                dataset_raw[lang_pair] = datasets.load_dataset(
                    name=hf_subset,
                    **self.metadata_dict["dataset"],
                )[split]

                queries[lang_pair] = {}
                corpus[lang_pair] = {}
                relevant_docs[lang_pair] = {}
                queries[lang_pair][split] = {}
                corpus[lang_pair][split] = {}
                relevant_docs[lang_pair][split] = {}

                # Generate unique IDs for queries and documents
                query_id_counter = 1
                document_id_counter = 1

                for row in dataset_raw[lang_pair]:
                    query_text = row["Source"]
                    positive_text = [row["Target"]]
                    negative_texts = [
                        row[f"TargetAdv{str(i)}"]
                        for i in range(
                            1, self.num_of_distractors + 1
                        )  # Four Distractors. Columns are named TargetAdv1-TargetAdv4
                    ]

                    # Assign unique ID to the query
                    query_id = f"Q{query_id_counter}"
                    queries[lang_pair][split][query_id] = query_text
                    query_id_counter += 1

                    # Add true parallel and distractors to corpus with unique id.
                    for text in positive_text + negative_texts:
                        doc_id = f"D{document_id_counter}"
                        corpus[lang_pair][split][doc_id] = {"text": text}
                        document_id_counter += 1

                        # Add relevant document information to relevant_docs for positive texts only
                        if text in positive_text:
                            if query_id not in relevant_docs[lang_pair][split]:
                                relevant_docs[lang_pair][split][query_id] = {}
                            relevant_docs[lang_pair][split][query_id][doc_id] = 1

            self.corpus = datasets.DatasetDict(corpus)
            self.queries = datasets.DatasetDict(queries)
            self.relevant_docs = datasets.DatasetDict(relevant_docs)

            self.data_loaded = True
