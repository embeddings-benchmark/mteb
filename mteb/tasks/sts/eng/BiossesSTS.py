from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class BiossesSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BIOSSES",
        dataset={
            "path": "mteb/biosses-sts",
            "revision": "d3fb88f8f02e40887cd149695127462bbcf29b4a",
        },
        description="Biomedical Semantic Similarity Estimation.",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Medical"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1093/bioinformatics/btx238,
  author = {Soğancıoğlu, Gizem and Öztürk, Hakime and Özgür, Arzucan},
  doi = {10.1093/bioinformatics/btx238},
  eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/14/i49/50315066/bioinformatics\_33\_14\_i49.pdf},
  issn = {1367-4803},
  journal = {Bioinformatics},
  month = {07},
  number = {14},
  pages = {i49-i58},
  title = {{BIOSSES: a semantic sentence similarity estimation system for the biomedical domain}},
  url = {https://doi.org/10.1093/bioinformatics/btx238},
  volume = {33},
  year = {2017},
}
""",
    )

    min_score = 0
    max_score = 5
