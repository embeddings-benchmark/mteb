from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

NQ_CITATION = r"""
@article{kwiatkowski_natural_2019,
  title = {Natural Questions: A Benchmark for Question Answering Research},
  author = {Kwiatkowski, Tom  and
      Palomaki, Jennimaria  and
      Redfield, Olivia  and
      Collins, Michael  and
      Parikh, Ankur  and
      Alberti, Chris  and
      Epstein, Danielle  and
      Polosukhin, Illia  and
      Devlin, Jacob  and
      Lee, Kenton  and
      Toutanova, Kristina  and
      Jones, Llion  and
      Kelcey, Matthew  and
      Chang, Ming-Wei  and
      Dai, Andrew M.  and
      Uszkoreit, Jakob  and
      Le, Quoc  and
      Petrov, Slav},
  editor = {Lee, Lillian  and
      Johnson, Mark  and
      Roark, Brian  and
      Nenkova, Ani},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {7},
  year = {2019},
  address = {Cambridge, MA},
  publisher = {MIT Press},
  url = {https://aclanthology.org/Q19-1026/},
  doi = {10.1162/tacl_a_00276},
  pages = {452--466},
}
"""


class NQ(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQ",
        dataset={
            "path": "mteb/nq",
            "revision": "b774495ed302d8c44a3a7ea25c90dbce03968f31",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),  # publication year
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=NQ_CITATION,
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
    )


class NQHardNegatives(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NQHardNegatives",
        dataset={
            "path": "mteb/NQ_test_top_250_only_w_correct-v2",
            "revision": "d700fe4f167a5db8e6c9b03e8c26e7eaf66faf97",
        },
        description="NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval. The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct.",
        reference="https://ai.google.com/research/NaturalQuestions/",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),  # publication year
        domains=["Written", "Encyclopaedic"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=NQ_CITATION,
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NQ"],
    )
