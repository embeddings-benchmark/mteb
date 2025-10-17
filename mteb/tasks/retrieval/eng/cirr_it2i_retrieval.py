from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class CIRRIT2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CIRRIT2IRetrieval",
        description="Retrieve images based on texts and images.",
        reference="https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html",
        dataset={
            "path": "MRBench/mbeir_cirr_task7",
            "revision": "503301cd99348035b9675883a543aa1ded0cf07c",
        },
        type="Any2AnyRetrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{liu2021image,
  author = {Liu, Zheyuan and Rodriguez-Opazo, Cristian and Teney, Damien and Gould, Stephen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages = {2125--2134},
  title = {Image retrieval on real-life images with pre-trained vision-and-language models},
  year = {2021},
}
""",
        prompt={
            "query": "Retrieve a day-to-day image that aligns with the modification instructions of the provided image."
        },
    )
