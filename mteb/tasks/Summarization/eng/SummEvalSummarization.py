from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSummarization


class SummEvalSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEval",
        description="News Article Summary Semantic Similarity Estimation.",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        dataset={
            "path": "mteb/summeval",
            "revision": "cda12ad7615edc362dbf25a00fdd61d3b1eaf93c",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@article{10.1093/bioinformatics/btx238,
    author = {Soğancıoğlu, Gizem and Öztürk, Hakime and Özgür, Arzucan},
    title = "{BIOSSES: a semantic sentence similarity estimation system for the biomedical domain}",
    journal = {Bioinformatics},
    volume = {33},
    number = {14},
    pages = {i49-i58},
    year = {2017},
    month = {07},
    abstract = "{The amount of information available in textual format is rapidly increasing in the biomedical domain. Therefore, natural language processing (NLP) applications are becoming increasingly important to facilitate the retrieval and analysis of these data. Computing the semantic similarity between sentences is an important component in many NLP tasks including text retrieval and summarization. A number of approaches have been proposed for semantic sentence similarity estimation for generic English. However, our experiments showed that such approaches do not effectively cover biomedical knowledge and produce poor results for biomedical text.We propose several approaches for sentence-level semantic similarity computation in the biomedical domain, including string similarity measures and measures based on the distributed vector representations of sentences learned in an unsupervised manner from a large biomedical corpus. In addition, ontology-based approaches are presented that utilize general and domain-specific ontologies. Finally, a supervised regression based model is developed that effectively combines the different similarity computation metrics. A benchmark data set consisting of 100 sentence pairs from the biomedical literature is manually annotated by five human experts and used for evaluating the proposed methods.The experiments showed that the supervised semantic sentence similarity computation approach obtained the best performance (0.836 correlation with gold standard human annotations) and improved over the state-of-the-art domain-independent systems up to 42.6\\% in terms of the Pearson correlation metric.A web-based system for biomedical semantic sentence similarity computation, the source code, and the annotated benchmark data set are available at: http://tabilab.cmpe.boun.edu.tr/BIOSSES/.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btx238},
    url = {https://doi.org/10.1093/bioinformatics/btx238},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/14/i49/50315066/bioinformatics\_33\_14\_i49.pdf},
}""",
        n_samples={"test": 2800},
        avg_character_length={"test": 359.8},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
