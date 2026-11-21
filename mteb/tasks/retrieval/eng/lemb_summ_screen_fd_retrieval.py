from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LEMBSummScreenFDRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "validation"

    metadata = TaskMetadata(
        name="LEMBSummScreenFDRetrieval",
        dataset={
            "path": "mteb/LEMBSummScreenFDRetrieval",
            "revision": "16ebb35b24c10f7d464a6869a5a88d04e6147e02",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("summ_screen_fd subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2021-12-31"),
        domains=["Spoken", "Written"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen-etal-2022-summscreen,
  address = {Dublin, Ireland},
  author = {Chen, Mingda  and
Chu, Zewei  and
Wiseman, Sam  and
Gimpel, Kevin},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2022.acl-long.589},
  editor = {Muresan, Smaranda  and
Nakov, Preslav  and
Villavicencio, Aline},
  month = may,
  pages = {8602--8615},
  publisher = {Association for Computational Linguistics},
  title = {{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization},
  url = {https://aclanthology.org/2022.acl-long.589},
  year = {2022},
}
""",
    )
