from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class MSCOCOI2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSCOCOI2TRetrieval",
        description="Retrieve captions based on images.",
        reference="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48",
        dataset={
            "path": "mteb/mbeir_mscoco_task3",
            "revision": "ec63d7122301171948387e757351fabd193fc39f",
        },
        type="Any2AnyRetrieval",
        category="i2t",
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
@inproceedings{lin_microsoft_2014,
  author = {Lin, Tsung-Yi
    and Maire, Michael
    and Belongie, Serge
    and Hays, James
    and Perona, Pietro
    and Ramanan, Deva
    and Doll{\'a}r, Piotr
    and Zitnick, C. Lawrence},
  editor = {Fleet, David
    and Pajdla, Tomas
    and Schiele, Bernt
    and Tuytelaars, Tinne},
  title = {Microsoft COCO: Common Objects in Context},
  booktitle = {Computer Vision -- ECCV 2014},
  year = {2014},
  publisher = {Springer International Publishing},
  address = {Cham},
  pages = {740--755},
  isbn = {978-3-319-10602-1},
}
""",
        prompt={
            "query": "Find an image caption describing the following everyday image."
        },
    )
