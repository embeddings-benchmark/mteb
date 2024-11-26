from __future__ import annotations

from hashlib import sha256

import datasets

from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_LANGUAGES = {
    "ar": ["arb-Arab"],
    "de": ["deu-Latn"],
    "el": ["ell-Grek"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "hi": ["hin-Deva"],
    "ro": ["ron-Latn"],
    "ru": ["rus-Cyrl"],
    "th": ["tha-Thai"],
    "tr": ["tur-Latn"],
    "vi": ["vie-Latn"],
    "zh": ["zho-Hans"],
}


class XQuADRetrieval(MultilingualTask, AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="XQuADRetrieval",
        dataset={
            "path": "google/xquad",
            "revision": "51adfef1c1287aab1d2d91b5bead9bcfb9c68583",
        },
        description="XQuAD is a benchmark dataset for evaluating cross-lingual question answering performance. It is repurposed retrieving relevant context for each question.",
        reference="https://huggingface.co/datasets/xquad",
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["validation"],
        eval_langs=_LANGUAGES,
        main_score="ndcg_at_10",
        date=("2019-05-21", "2019-11-21"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
@inproceedings{
      dumitrescu2021liro,
      title={LiRo: Benchmark and leaderboard for Romanian language tasks},
      author={Stefan Daniel Dumitrescu and Petru Rebeja and Beata Lorincz and Mihaela Gaman and Andrei Avram and Mihai Ilie and Andrei Pruteanu and Adriana Stan and Lorena Rosia and Cristina Iacobescu and Luciana Morogan and George Dima and Gabriel Marchidan and Traian Rebedea and Madalina Chitez and Dani Yogatama and Sebastian Ruder and Radu Tudor Ionescu and Razvan Pascanu and Viorica Patraucean},
      booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
      year={2021},
      url={https://openreview.net/forum?id=JH61CD7afTv}
}""",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        split = "validation"
        queries = {lang: {split: {}} for lang in self.hf_subsets}
        corpus = {lang: {split: {}} for lang in self.hf_subsets}
        relevant_docs = {lang: {split: {}} for lang in self.hf_subsets}

        for lang in self.hf_subsets:
            data = datasets.load_dataset(
                name=f"xquad.{lang}", **self.metadata_dict["dataset"]
            )[split]
            data = data.filter(lambda x: x["answers"]["text"] != "")

            question_ids = {
                question: id for id, question in zip(data["id"], data["question"])
            }
            context_ids = {
                context: sha256(context.encode("utf-8")).hexdigest()
                for context in set(data["context"])
            }

            for row in data:
                question = row["question"]
                context = row["context"]
                query_id = question_ids[question]
                queries[lang][split][query_id] = question

                doc_id = context_ids[context]
                corpus[lang][split][doc_id] = {"text": context}
                if query_id not in relevant_docs[lang][split]:
                    relevant_docs[lang][split][query_id] = {}
                relevant_docs[lang][split][query_id][doc_id] = 1

        self.corpus = datasets.DatasetDict(corpus)
        self.queries = datasets.DatasetDict(queries)
        self.relevant_docs = datasets.DatasetDict(relevant_docs)

        self.data_loaded = True
