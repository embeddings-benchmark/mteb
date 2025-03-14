from __future__ import annotations

import itertools

from datasets import Dataset, DatasetDict

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import (
    AbsTaskClusteringFast,
    check_label_distribution,
)
from mteb.abstasks.TaskMetadata import TaskMetadata


class ClusTrecCovidP2PFast(AbsTaskClusteringFast):
    metadata = TaskMetadata(
        name="ClusTREC-CovidP2P.v2",
        description="A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.",
        reference="https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID",
        dataset={
            "path": "Uri-ka/ClusTREC-Covid",
            "revision": "7f3489153b8dad7336a54f63202deb1414c33309",
        },
        type="Clustering",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        HFSubset="title and abstract",
        eval_langs={
            "title and abstract": ["eng-Latn"],
            "title": ["eng-Latn"]
        },
        main_score="v_measure",
        date=("2020-04-10", "2020-07-16"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation="""@inproceedings{katz-etal-2024-knowledge,
            title = "Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature",
            author = "Katz, Uri  and
              Levy, Mosh  and
              Goldberg, Yoav",
            booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
            month = nov,
            year = "2024",
            address = "Miami, Florida, USA",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2024.findings-emnlp.516",
            pages = "8838--8855",
        }
                """,
        prompt="Identify the main category of the covid-19 papers based on the titles and abstracts",
        adapted_from=["ClusTrecCovidP2P"],
    )


# class ClusTrecCovidP2P(AbsTaskClustering):
#     metadata = TaskMetadata(
#         name="ClusTREC-CovidP2P",
#         description="A Topical Clustering Benchmark for COVID-19 Scientific Research across 50 covid-19 related topics.",
#         reference="https://github.com/katzurik/Knowledge_Navigator/tree/main/Benchmarks/CLUSTREC%20COVID",
#         dataset={
#             "path": "Uri-ka/ClusTREC-Covid-P2P",
#             "revision": "f5a9420d2bd39d472ca0d3c268ea88d7e9a10fa4",
#         },
#         type="Clustering",
#         category="p2p",
#         modalities=["text"],
#         eval_splits=["test"],
#         eval_langs={
#             "title and abstract": ["eng-Latn"],
#             "title": ["eng-Latn"]
#         },
#         main_score="v_measure",
#         date=("2020-04-10", "2020-07-16"),
#         domains=["Academic", "Written"],
#         task_subtypes=["Thematic clustering"],
#         license="cc-by-sa-4.0",
#         annotations_creators="expert-annotated",
#         dialect=[],
#         sample_creation="created",
#         bibtex_citation="""@inproceedings{katz-etal-2024-knowledge,
#     title = "Knowledge Navigator: {LLM}-guided Browsing Framework for Exploratory Search in Scientific Literature",
#     author = "Katz, Uri  and
#       Levy, Mosh  and
#       Goldberg, Yoav",
#     booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
#     month = nov,
#     year = "2024",
#     address = "Miami, Florida, USA",
#     publisher = "Association for Computational Linguistics",
#     url = "https://aclanthology.org/2024.findings-emnlp.516",
#     pages = "8838--8855",
# }
#         """,
#         prompt="Identify the main category of the covid-19 papers based on the titles and abstracts",
#     )

if __name__ == "__main__":
    # testing the task with a model:
    from sentence_transformers import SentenceTransformer
    from mteb.evaluation.MTEB import MTEB

    model = SentenceTransformer("average_word_embeddings_komninos")
    evaluation = MTEB(tasks=[ClusTrecCovidP2PFast()])
    evaluation.run(model)