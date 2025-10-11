from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.retrieval import AbsTaskRetrieval


class Fashion200kI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Fashion200kI2TRetrieval",
        description="Retrieve clothes based on descriptions.",
        reference="https://openaccess.thecvf.com/content_iccv_2017/html/Han_Automatic_Spatially-Aware_Fashion_ICCV_2017_paper.html",
        dataset={
            "path": "MRBench/mbeir_fashion200k_task3",
            "revision": "96a313715ecf67f5dfe70c4fa52406bc7bdfbeee",
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{han2017automatic,
  author = {Han, Xintong and Wu, Zuxuan and Huang, Phoenix X and Zhang, Xiao and Zhu, Menglong and Li, Yuan and Zhao, Yang and Davis, Larry S},
  booktitle = {Proceedings of the IEEE international conference on computer vision},
  pages = {1463--1471},
  title = {Automatic spatially-aware fashion concept discovery},
  year = {2017},
}
""",
        prompt={
            "query": "Based on the following fashion description, retrieve the best matching image."
        },
    )
