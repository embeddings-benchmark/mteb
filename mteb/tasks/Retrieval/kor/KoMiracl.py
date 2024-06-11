from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class KoMiracl(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Ko-miracl",
        description="Ko-miracl",
        reference=None,
        dataset={
            "path": "taeminlee/Ko-miracl",
            "revision": "5c7690518e481375551916f24241048cf7b017d0",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["dev"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@article{10.1162/tacl_a_00595,
    author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
    title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1114-1131},
    year = {2023},
    month = {09},
    abstract = "{MIRACL is a multilingual dataset for ad hoc retrieval across 18 languages that collectively encompass over three billion native speakers around the world. This resource is designed to support monolingual retrieval tasks, where the queries and the corpora are in the same language. In total, we have gathered over 726k high-quality relevance judgments for 78k queries over Wikipedia in these languages, where all annotations have been performed by native speakers hired by our team. MIRACL covers languages that are both typologically close as well as distant from 10 language families and 13 sub-families, associated with varying amounts of publicly available resources. Extensive automatic heuristic verification and manual assessments were performed during the annotation process to control data quality. In total, MIRACL represents an investment of around five person-years of human annotator effort. Our goal is to spur research on improving retrieval across a continuum of languages, thus enhancing information access capabilities for diverse populations around the world, particularly those that have traditionally been underserved. MIRACL is available at http://miracl.ai/.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00595},
    url = {https://doi.org/10.1162/tacl\_a\_00595},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
}""",
        n_samples=None,
        avg_character_length=None,
    )
