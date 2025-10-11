from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLaVAIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLaVAIT2TRetrieval",
        description="Retrieve responses to answer questions about images.",
        reference="https://github.com/LinWeizheDragon/FLMR/blob/main/docs/Datasets.md",
        dataset={
            "path": "izhx/UMRB-LLaVA",
            "revision": "2a5ed414aab388d8cdd244ba2cf8c8960df4d44d",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cv_recall_at_5",
        date=("2024-07-06", "2024-02-26"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{lin-etal-2024-preflmr,
  address = {Bangkok, Thailand},
  author = {Lin, Weizhe  and
Mei, Jingbiao  and
Chen, Jinghong  and
Byrne, Bill},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2024.acl-long.289},
  editor = {Ku, Lun-Wei  and
Martins, Andre  and
Srikumar, Vivek},
  month = aug,
  pages = {5294--5316},
  publisher = {Association for Computational Linguistics},
  title = {{P}re{FLMR}: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers},
  url = {https://aclanthology.org/2024.acl-long.289},
  year = {2024},
}
""",
        prompt={
            "query": "Provide a specific decription of the image along with the following question."
        },
    )
