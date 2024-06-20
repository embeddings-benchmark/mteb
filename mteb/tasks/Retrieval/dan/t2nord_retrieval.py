import datasets

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


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
        category="p2p",
        eval_splits=["test"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2024-12-31"),  # best guess
        form=["written"],
        domains=["News", "Non-fiction"],
        license="CC0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=[],
        text_creation="found",
        bibtex_citation="""@inproceedings{flansmose-mikkelsen-etal-2022-ddisco,
    title = "{DD}is{C}o: A Discourse Coherence Dataset for {D}anish",
    author = "Flansmose Mikkelsen, Linea  and
      Kinch, Oliver  and
      Jess Pedersen, Anders  and
      Lacroix, Oph{\'e}lie",
    editor = "Calzolari, Nicoletta  and
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
      Piperidis, Stelios",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.260",
    pages = "2440--2445",
    abstract = "To date, there has been no resource for studying discourse coherence on real-world Danish texts. Discourse coherence has mostly been approached with the assumption that incoherent texts can be represented by coherent texts in which sentences have been shuffled. However, incoherent real-world texts rarely resemble that. We thus present DDisCo, a dataset including text from the Danish Wikipedia and Reddit annotated for discourse coherence. We choose to annotate real-world texts instead of relying on artificially incoherent text for training and testing models. Then, we evaluate the performance of several methods, including neural networks, on the dataset.",
}""",
        n_samples={"test": 4096},
        avg_character_length={"test": 784.11},
        task_subtypes=["Article retrieval"],
    )

    def load_data(self, **kwargs):
        """Load dataset from HuggingFace hub"""
        if self.data_loaded:
            return
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self) -> None:
        """And transform to a retrieval datset, which have the following attributes

        self.corpus = Dict[doc_id, Dict[str, str]] #id => dict with document datas like title and text
        self.queries = Dict[query_id, str] #id => query
        self.relevant_docs = Dict[query_id, Dict[[doc_id, score]]
        """
        self.corpus = {}
        self.relevant_docs = {}
        self.queries = {}
        text2id = {}

        for split in self.dataset:
            ds: datasets.Dataset = self.dataset[split]  # type: ignore
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
