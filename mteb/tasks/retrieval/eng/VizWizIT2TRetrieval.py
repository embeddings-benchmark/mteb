from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class VizWizIT2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VizWizIT2TRetrieval",
        description="Retrieve the correct answer for a question about an image.",
        reference="https://openaccess.thecvf.com/content_cvpr_2018/papers/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.pdf",
        dataset={
            "path": "JamieSJS/vizwiz",
            "revision": "044af162d55f82ab603fa16ffcf7f1e4dbf300e9",
        },
        type="Any2AnyRetrieval",
        category="it2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-01-01"),
        domains=["Web"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{gurari2018vizwiz,
  author = {Gurari, Danna and Li, Qing and Stangl, Abigale J and Guo, Anhong and Lin, Chi and Grauman, Kristen and Luo, Jiebo and Bigham, Jeffrey P},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages = {3608--3617},
  title = {Vizwiz grand challenge: Answering visual questions from blind people},
  year = {2018},
}
""",
    )
