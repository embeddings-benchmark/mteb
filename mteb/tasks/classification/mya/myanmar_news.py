from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class MyanmarNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MyanmarNews",
        dataset={
            "path": "mteb/MyanmarNews",
            "revision": "644419f24bc820bbf8af24e0b4714a069812e0a3",
        },
        description="The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categories, providing a rich resource for natural language processing applications involving Burmese which is a low resource language.",
        reference="https://huggingface.co/datasets/myanmar_news",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["mya-Mymr"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Khine2017,
  author = {A. H. Khine and K. T. Nwet and K. M. Soe},
  booktitle = {15th Proceedings of International Conference on Computer Applications},
  month = {February},
  pages = {401--408},
  title = {Automatic Myanmar News Classification},
  year = {2017},
}
""",
        superseded_by="MyanmarNews.v2",
    )


class MyanmarNewsV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MyanmarNews.v2",
        dataset={
            "path": "mteb/myanmar_news",
            "revision": "475b43ffbdb5138ad67a01a2c860bc7db502f3c5",
        },
        description="The Myanmar News dataset on Hugging Face contains news articles in Burmese. It is designed for tasks such as text classification, sentiment analysis, and language modeling. The dataset includes a variety of news topics in 4 categories, providing a rich resource for natural language processing applications involving Burmese which is a low resource language. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/myanmar_news",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["mya-Mymr"],
        main_score="accuracy",
        date=("2017-10-01", "2017-10-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="gpl-3.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Khine2017,
  author = {A. H. Khine and K. T. Nwet and K. M. Soe},
  booktitle = {15th Proceedings of International Conference on Computer Applications},
  month = {February},
  pages = {401--408},
  title = {Automatic Myanmar News Classification},
  year = {2017},
}
""",
        adapted_from=["MyanmarNews"],
    )
