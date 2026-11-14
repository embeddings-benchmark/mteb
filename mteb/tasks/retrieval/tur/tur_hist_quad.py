from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TurHistQuadRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TurHistQuadRetrieval",
        dataset={
            "path": "mteb/TurHistQuadRetrieval",
            "revision": "b6e74379b7486da28ce81c3d459cd7bbd87d4987",
        },
        description="Question Answering dataset on Ottoman History in Turkish",
        reference="https://github.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["tur-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-10-13"),
        task_subtypes=["Question answering"],
        domains=["Encyclopaedic", "Non-fiction", "Academic", "Written"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{9559013,
  author = {Soygazi, Fatih and Çiftçi, Okan and Kök, Uğurcan and Cengiz, Soner},
  booktitle = {2021 6th International Conference on Computer Science and Engineering (UBMK)},
  doi = {10.1109/UBMK52708.2021.9559013},
  keywords = {Computer science;Computational modeling;Neural networks;Knowledge discovery;Information retrieval;Natural language processing;History;question answering;information retrieval;natural language understanding;deep learning;contextualized word embeddings},
  number = {},
  pages = {215-220},
  title = {THQuAD: Turkish Historic Question Answering Dataset for Reading Comprehension},
  volume = {},
  year = {2021},
}
""",
    )
