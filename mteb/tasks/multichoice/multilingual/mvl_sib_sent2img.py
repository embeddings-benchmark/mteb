from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.tasks.classification.multilingual.sib200_classification import (
    SIB200Classification,
)

_EXTRA_LANGS = {
    "ace_Arab": ["ace-Arab"],
    "arb_Arab": ["arb-Arab"],
    "bjn_Arab": ["bjn-Arab"],
    "kas_Arab": ["kas-Arab"],
    "knc_Arab": ["knc-Arab"],
    "min_Arab": ["min-Arab"],
    "taq_Latn": ["taq-Latn"],
    "zho_Hans": ["zho-Hans"],
}
_LANGUAGES = {
    **SIB200Classification.metadata.hf_subsets_to_langscripts,
    **_EXTRA_LANGS,
}


class MVLSIBSent2Img(AbsTaskRetrieval):
    """MVL-SIB single-reference sentence-to-image benchmark."""

    k_values = (1, 2, 3, 4)

    metadata = TaskMetadata(
        name="MVLSIBSent2Img",
        description=(
            "Choose the topically matching image from four candidates for one "
            "reference sentence. This is the paper's single-reference (k=1) "
            "sentence-to-image setting in all 205 SIB-200 languages and scripts."
        ),
        reference="https://aclanthology.org/2025.findings-acl.838/",
        dataset={
            "path": "artist/mvl-sib-sent2img-mteb",
            "revision": "3323e2ec3edfaed11c2802b6640dd61091c3755e",
        },
        type="VisionCentricQA",
        category="t2i",
        modalities=["text", "image"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="accuracy",
        date=("2025-02-18", "2025-02-18"),
        domains=["News", "Scene", "Written"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="multiple",
        adapted_from=["SIB200Classification"],
        is_beta=True,
        bibtex_citation=r"""
@inproceedings{schmidt-etal-2025-mvl,
  address = {Vienna, Austria},
  author = {Fabian David Schmidt and Florian Schneider and Chris Biemann and Goran Glava{\v{s}}},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  doi = {10.18653/v1/2025.findings-acl.838},
  month = jul,
  pages = {16285--16312},
  publisher = {Association for Computational Linguistics},
  title = {{MVL-SIB}: A Massively Multilingual Vision-Language Benchmark for Cross-Modal Topical Matching},
  url = {https://aclanthology.org/2025.findings-acl.838/},
  year = {2025},
}
""",
        prompt={"query": "Find the image matching this reference sentence."},
    )
