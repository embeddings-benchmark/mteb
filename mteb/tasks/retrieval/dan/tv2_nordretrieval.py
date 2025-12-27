import datasets

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class TV2Nordretrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TV2Nordretrieval",
        dataset={
            "path": "alexandrainst/nordjylland-news-summarization",
            "revision": "80cdb115ec2ef46d4e926b252f2b59af62d6c070",
        },
        description="News Article and corresponding summaries extracted from the Danish newspaper TV2 Nord.",
        reference="https://huggingface.co/datasets/alexandrainst/nordjylland-news-summarization",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        domains=["News", "Non-fiction", "Written"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
  address = {Marseille, France},
  author = {Flansmose Mikkelsen, Linea  and
Kinch, Oliver  and
Jess Pedersen, Anders  and
Lacroix, Oph{\'e}lie},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  editor = {Calzolari, Nicoletta  and
B{\'e}chet, Fr{\'e}d{\'e}ric  and
Blache, Philippe  and
Choukri, Khalid  and
Cieri, Christopher  and
Declerck, Thierry  and
Goggi, Sara  and
Isahara, Hitoshi  and
Maegaard, Bente  and
Mariani, Joseph  and
Mazo, H{\'e}l{\`e}ne  and
Odijk, Jan  and
Piperidis, Stelios},
  month = jun,
  pages = {2440--2445},
  publisher = {European Language Resources Association},
  title = {{DD}is{C}o: A Discourse Coherence Dataset for {D}anish},
  url = {https://aclanthology.org/2022.lrec-1.260},
  year = {2022},
}
""",
        prompt={
            "query": "Given a summary of a Danish news article retrieve the corresponding news article"
        },
        task_subtypes=["Article retrieval"],
    )

    def load_data(self) -> None:
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval dataset, which have the following attributes

        self.corpus = dict[doc_id, dict[str, str]] #id => dict with document data like title and text
        self.queries = dict[query_id, str] #id => query
        self.relevant_docs = dict[query_id, dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]
            ds = ds.shuffle(seed=42)
            ds = ds.select(
                range(2048)
            )  # limit the dataset size to make sure the task does not take too long to run
            self.queries[split] = {}
            self.relevant_docs[split] = {}
            self.corpus[split] = {}

            summary = ds["summary"]
            article = ds["text"]

            n = 0
            for summ, art in zip(summary, article):
                self.queries[split][str(n)] = summ
                q_n = n
                n += 1
                if art not in text2id:
                    text2id[art] = n
                    self.corpus[split][str(n)] = {"title": "", "text": art}
                    n += 1

                self.relevant_docs[split][str(q_n)] = {
                    str(text2id[art]): 1
                }  # only one correct matches
