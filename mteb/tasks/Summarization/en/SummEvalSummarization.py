from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSummarization


class SummEvalSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="SummEval",
        description="News Article Summary Semantic Similarity Estimation.",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        hf_hub_name="mteb/summeval",
        type="Summarization",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="cosine_spearman",
        revision="cda12ad7615edc362dbf25a00fdd61d3b1eaf93c",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = dict(self.metadata)
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
