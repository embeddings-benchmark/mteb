from __future__ import annotations

from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.TaskMetadata import TaskMetadata


class MewsC16JaClustering(AbsTaskClusteringFast):
    max_document_to_embed = 992
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="MewsC16JaClustering",
        description="""MewsC-16 (Multilingual Short Text Clustering Dataset for News in 16 languages) is constructed from Wikinews.
        This dataset is the Japanese split of MewsC-16, containing topic sentences from Wikinews articles in 12 categories.
        More detailed information is available in the Appendix E of the citation.""",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "sbintuitions/JMTEB",
            "name": "mewsc16_ja",
            "revision": "e4af6c73182bebb41d94cb336846e5a452454ea7",
            "trust_remote_code": True,
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["jpn-Jpan"],
        main_score="v_measure",
        date=("2002-01-01", "2019-01-31"),
        domains=["News", "Written"],
        task_subtypes=["Topic classification"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""
        @inproceedings{
            nishikawa-etal-2022-ease,
            title = "{EASE}: Entity-Aware Contrastive Learning of Sentence Embedding",
            author = "Nishikawa, Sosuke  and
            Ri, Ryokan  and
            Yamada, Ikuya  and
            Tsuruoka, Yoshimasa  and
            Echizen, Isao",
            booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
            month = jul,
            year = "2022",
            address = "Seattle, United States",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2022.naacl-main.284",
            pages = "3870--3885",
            abstract = "We present EASE, a novel method for learning sentence embeddings via contrastive learning between sentences and their related entities.The advantage of using entity supervision is twofold: (1) entities have been shown to be a strong indicator of text semantics and thus should provide rich training signals for sentence embeddings; (2) entities are defined independently of languages and thus offer useful cross-lingual alignment supervision.We evaluate EASE against other unsupervised models both in monolingual and multilingual settings.We show that EASE exhibits competitive or better performance in English semantic textual similarity (STS) and short text clustering (STC) tasks and it significantly outperforms baseline methods in multilingual settings on a variety of tasks.Our source code, pre-trained models, and newly constructed multi-lingual STC dataset are available at https://github.com/studio-ousia/ease.",
        }
        """,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns(
            {"text": "sentences", "label": "labels"}
        )
