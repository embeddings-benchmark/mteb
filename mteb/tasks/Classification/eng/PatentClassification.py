from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PatentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PatentClassification",
        description="Classification Dataset of Patents and Abstract",
        dataset={
            "path": "mteb/PatentClassification",
            "revision": "6bd77eb030ab3bfbf1e6f7a2b069979daf167311",
        },
        reference="https://aclanthology.org/P19-1212.pdf",
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-11-05", "2022-10-22"),
        domains=["Legal", "Written"],
        task_subtypes=["Topic classification"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{sharma-etal-2019-bigpatent,
  abstract = {Most existing text summarization datasets are compiled from the news domain, where summaries have a flattened discourse structure. In such datasets, summary-worthy content often appears in the beginning of input articles. Moreover, large segments from input articles are present verbatim in their respective summaries. These issues impede the learning and evaluation of systems that can understand an article{'}s global content structure as well as produce abstractive summaries with high compression ratio. In this work, we present a novel dataset, BIGPATENT, consisting of 1.3 million records of U.S. patent documents along with human written abstractive summaries. Compared to existing summarization datasets, BIGPATENT has the following properties: i) summaries contain a richer discourse structure with more recurring entities, ii) salient content is evenly distributed in the input, and iii) lesser and shorter extractive fragments are present in the summaries. Finally, we train and evaluate baselines and popular learning models on BIGPATENT to shed light on new challenges and motivate future directions for summarization research.},
  address = {Florence, Italy},
  author = {Sharma, Eva  and
Li, Chen  and
Wang, Lu},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/P19-1212},
  editor = {Korhonen, Anna  and
Traum, David  and
M{\`a}rquez, Llu{\'\i}s},
  month = jul,
  pages = {2204--2213},
  publisher = {Association for Computational Linguistics},
  title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
  url = {https://aclanthology.org/P19-1212},
  year = {2019},
}
""",
    )
