from __future__ import annotations

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CIRRIT2IRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="CIRRIT2IRetrieval",
        description="Retrieve images based on texts and images.",
        reference="https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html",
        dataset={
            "path": "MRBench/mbeir_cirr_task7",
            "revision": "503301cd99348035b9675883a543aa1ded0cf07c",
            "trust_remote_code": True,
        },
        type="Retrieval",
        category="it2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        form=["written"],
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="CC BY-SA 4.0",
        socioeconomic_status="medium",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation="""@inproceedings{liu2021image,
        title={Image retrieval on real-life images with pre-trained vision-and-language models},
        author={Liu, Zheyuan and Rodriguez-Opazo, Cristian and Teney, Damien and Gould, Stephen},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={2125--2134},
        year={2021}
        }""",
        descriptive_stats={
            "n_samples": {"test": 1172},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 9350,
                    "num_queries": 1172,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
