from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class MewsC16JaClustering(AbsTaskClustering):
    max_document_to_embed = 992
    max_fraction_of_documents_to_embed = None

    metadata = TaskMetadata(
        name="MewsC16JaClustering",
        description="MewsC-16 (Multilingual Short Text Clustering Dataset for News in 16 languages) is constructed from Wikinews. This dataset is the Japanese split of MewsC-16, containing topic sentences from Wikinews articles in 12 categories. More detailed information is available in the Appendix E of the citation.",
        reference="https://github.com/sbintuitions/JMTEB",
        dataset={
            "path": "mteb/MewsC16JaClustering",
            "revision": "f50ef5a569175d8a1a04cbbe2c8012d138eb5e71",
        },
        type="Clustering",
        category="t2c",
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
        bibtex_citation=r"""
@inproceedings{nishikawa-etal-2022-ease,
  address = {Seattle, United States},
  author = {Nishikawa, Sosuke  and
Ri, Ryokan  and
Yamada, Ikuya  and
Tsuruoka, Yoshimasa  and
Echizen, Isao},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  month = jul,
  pages = {3870--3885},
  publisher = {Association for Computational Linguistics},
  title = {{EASE}: Entity-Aware Contrastive Learning of Sentence Embedding},
  url = {https://aclanthology.org/2022.naacl-main.284},
  year = {2022},
}
""",
    )
