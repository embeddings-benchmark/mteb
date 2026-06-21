from collections import defaultdict

from datasets import Value, load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "ar": ["ara-Arab"],
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
    "ja": ["jpn-Jpan"],
    "ko": ["kor-Kore"],
    "no": ["nor-Latn"],
    "pt": ["por-Latn"],
    "sv": ["swe-Latn"],
}


class AbsTaskNanoBEIRMultilingual(AbsTaskRetrieval):
    task_base_name: str

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return

        self.dataset = {}
        for lang in self.hf_subsets:
            ds = load_dataset(
                self.metadata.dataset["path"],
                f"{self.task_base_name}_{lang}",
                revision=self.metadata.dataset["revision"],
            )
            corpus = ds["corpus"]
            queries = ds["queries"]

            corpus = corpus.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )
            queries = queries.cast_column("_id", Value("string")).rename_column(
                "_id", "id"
            )

            qrels_ds = load_dataset(
                self.metadata.dataset["path"],
                f"{self.task_base_name}_{lang}_qrels",
                revision=self.metadata.dataset["revision"],
            )["qrels"]

            relevant_docs = defaultdict(dict)
            for row in qrels_ds:
                relevant_docs[str(row["query-id"])][str(row["corpus-id"])] = 1

            self.dataset[lang] = {
                "test": RetrievalSplitData(
                    corpus=corpus,
                    queries=queries,
                    relevant_docs=relevant_docs,
                    top_ranked=None,
                )
            }

        self.data_loaded = True


class MultilingualNanoArguAnaRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoArguAna"

    metadata = TaskMetadata(
        name="MultilingualNanoArguAnaRetrieval",
        description="NanoArguAna is a smaller subset of ArguAna, a dataset for argument retrieval in debate contexts.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{wachsmuth2018retrieval,
  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
  booktitle = {ACL},
  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
  year = {2018},
}
""",
        prompt={"query": "Given a claim, find documents that refute the claim"},
        adapted_from=["NanoArguAnaRetrieval"],
    )


class MultilingualNanoClimateFeverRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoClimateFEVER"

    metadata = TaskMetadata(
        name="MultilingualNanoClimateFeverRetrieval",
        description="NanoClimateFever is a small version of the BEIR dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Non-fiction", "Academic", "News"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{diggelmann2021climatefever,
  archiveprefix = {arXiv},
  author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
  eprint = {2012.00614},
  primaryclass = {cs.CL},
  title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
        adapted_from=["NanoClimateFeverRetrieval"],
    )


class MultilingualNanoDBPediaRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoDBPedia"

    metadata = TaskMetadata(
        name="MultilingualNanoDBPediaRetrieval",
        description="NanoDBPediaRetrieval is a small version of the standard test collection for entity search over the DBpedia knowledge base.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2015-01-01", "2015-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Topic classification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{lehmann2015dbpedia,
  author = {Lehmann, Jens and et al.},
  journal = {Semantic Web},
  title = {DBpedia: A large-scale, multilingual knowledge base extracted from Wikipedia},
  year = {2015},
}
""",
        prompt={
            "query": "Given a query, retrieve relevant entity descriptions from DBPedia"
        },
        adapted_from=["NanoDBPediaRetrieval"],
    )


class MultilingualNanoFEVERRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoFEVER"

    metadata = TaskMetadata(
        name="MultilingualNanoFEVERRetrieval",
        description="NanoFEVER is a smaller version of FEVER (Fact Extraction and VERification), which consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Encyclopaedic"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{thorne-etal-2018-fever,
  address = {New Orleans, Louisiana},
  author = {Thorne, James  and Vlachos, Andreas  and Christodoulopoulos, Christos  and Mittal, Arpit},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  doi = {10.18653/v1/N18-1074},
  editor = {Walker, Marilyn  and Ji, Heng  and Stent, Amanda},
  month = jun,
  pages = {809--819},
  publisher = {Association for Computational Linguistics},
  title = {{FEVER}: a Large-scale Dataset for Fact Extraction and {VER}ification},
  url = {https://aclanthology.org/N18-1074},
  year = {2018},
}
""",
        prompt={
            "query": "Given a claim, retrieve documents that support or refute the claim"
        },
        adapted_from=["NanoFEVERRetrieval"],
    )


class MultilingualNanoFiQA2018Retrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoFiQA2018"

    metadata = TaskMetadata(
        name="MultilingualNanoFiQA2018Retrieval",
        description="NanoFiQA2018 is a smaller subset of the Financial Opinion Mining and Question Answering dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{thakur2021beir,
  author = {Nandan Thakur and Nils Reimers and Andreas R{"u}ckl'e and Abhishek Srivastava and Iryna Gurevych},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  title = {{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  url = {https://openreview.net/forum?id=wCu6T5xFjeJ},
  year = {2021},
}
""",
        prompt={
            "query": "Given a financial question, retrieve user replies that best answer the question"
        },
        adapted_from=["NanoFiQA2018Retrieval"],
    )


class MultilingualNanoHotpotQARetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoHotpotQA"

    metadata = TaskMetadata(
        name="MultilingualNanoHotpotQARetrieval",
        description="NanoHotpotQARetrieval is a smaller subset of the HotpotQA dataset, which is a question answering dataset featuring natural, multi-hop questions, with strong supervision for supporting facts to enable more explainable question answering systems.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{yang-etal-2018-hotpotqa,
  address = {Brussels, Belgium},
  author = {Yang, Zhilin  and
Qi, Peng  and
Zhang, Saizheng  and
Bengio, Yoshua  and
Cohen, William  and
Salakhutdinov, Ruslan  and
Manning, Christopher D.},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1259},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {2369--2380},
  publisher = {Association for Computational Linguistics},
  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  url = {https://aclanthology.org/D18-1259},
  year = {2018},
}
""",
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
        adapted_from=["NanoHotpotQARetrieval"],
    )


class MultilingualNanoMSMARCORetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoMSMARCO"

    metadata = TaskMetadata(
        name="MultilingualNanoMSMARCORetrieval",
        description="NanoMSMARCORetrieval is a smaller subset of MS MARCO, a collection of datasets focused on deep learning in search.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-31"),
        domains=["Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{DBLP:journals/corr/NguyenRSGTMD16,
  archiveprefix = {arXiv},
  author = {Tri Nguyen and Mir Rosenberg and Xia Song and Jianfeng Gao and Saurabh Tiwary and Rangan Majumder and Li Deng},
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
  eprint = {1611.09268},
  journal = {CoRR},
  timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
  title = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
  url = {http://arxiv.org/abs/1611.09268},
  volume = {abs/1611.09268},
  year = {2016},
}
""",
        prompt={
            "query": "Given a web search query, retrieve relevant passages that answer the query"
        },
        adapted_from=["NanoMSMARCORetrieval"],
    )


class MultilingualNanoNFCorpusRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoNFCorpus"

    metadata = TaskMetadata(
        name="MultilingualNanoNFCorpusRetrieval",
        description="NanoNFCorpus is a smaller subset of NFCorpus: A Full-Text Learning to Rank Dataset for Medical Information Retrieval.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2016-01-01", "2016-12-31"),
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{boteva2016,
  author = {Boteva, Vera and Gholipour, Demian and Sokolov, Artem and Riezler, Stefan},
  city = {Padova},
  country = {Italy},
  journal = {Proceedings of the 38th European Conference on Information Retrieval},
  journal-abbrev = {ECIR},
  title = {A Full-Text Learning to Rank Dataset for Medical Information Retrieval},
  url = {http://www.cl.uni-heidelberg.de/~riezler/publications/papers/ECIR2016.pdf},
  year = {2016},
}
""",
        prompt={
            "query": "Given a question, retrieve relevant documents that best answer the question"
        },
        adapted_from=["NanoNFCorpusRetrieval"],
    )


class MultilingualNanoNQRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoNQ"

    metadata = TaskMetadata(
        name="MultilingualNanoNQRetrieval",
        description="NanoNQ is a smaller subset of a dataset which contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=["Academic", "Web"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@article{47761,
  author = {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
  journal = {Transactions of the Association of Computational Linguistics},
  title = {Natural Questions: a Benchmark for Question Answering Research},
  year = {2019},
}
""",
        prompt={
            "query": "Given a question, retrieve Wikipedia passages that answer the question"
        },
        adapted_from=["NanoNQRetrieval"],
    )


class MultilingualNanoQuoraRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoQuoraRetrieval"

    metadata = TaskMetadata(
        name="MultilingualNanoQuoraRetrieval",
        description="NanoQuoraRetrieval is a smaller subset of the QuoraRetrieval dataset, which is based on questions that are marked as duplicates on the Quora platform. Given a question, find other (duplicate) questions.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2017-01-01", "2017-12-31"),
        domains=["Social"],
        task_subtypes=["Duplicate Detection"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@misc{quora-question-pairs,
  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
  publisher = {Kaggle},
  title = {Quora Question Pairs},
  url = {https://kaggle.com/competitions/quora-question-pairs},
  year = {2017},
}
""",
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        adapted_from=["NanoQuoraRetrieval"],
    )


class MultilingualNanoSCIDOCSRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoSCIDOCS"

    metadata = TaskMetadata(
        name="MultilingualNanoSCIDOCSRetrieval",
        description="NanoFiQA2018 is a smaller subset of SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation prediction, to document classification and recommendation.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Academic", "Written", "Non-fiction"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        prompt={
            "query": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
        },
        adapted_from=["NanoSCIDOCSRetrieval"],
    )


class MultilingualNanoSciFactRetrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoSciFact"

    metadata = TaskMetadata(
        name="MultilingualNanoSciFactRetrieval",
        description="NanoSciFact is a smaller subset of SciFact, which verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2018-01-01", "2018-12-31"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
        adapted_from=["NanoSciFactRetrieval"],
    )


class MultilingualNanoTouche2020Retrieval(AbsTaskNanoBEIRMultilingual):
    task_base_name = "NanoTouche2020"

    metadata = TaskMetadata(
        name="MultilingualNanoTouche2020Retrieval",
        description="NanoTouche2020 is a smaller subset of Touché Task 1: Argument Retrieval for Controversial Questions.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "LiquidAI/nanobeir-multilingual-extended",
            "revision": "8a4be55eb80b3ed4d2e9a423a5212228c217d426",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated",
        bibtex_citation=r"""
@dataset{potthast_2022_6862281,
  author = {Potthast, Martin and
Gienapp, Lukas and
Wachsmuth, Henning and
Hagen, Matthias and
Fröbe, Maik and
Bondarenko, Alexander and
Ajjour, Yamen and
Stein, Benno},
  doi = {10.5281/zenodo.6862281},
  month = jul,
  publisher = {Zenodo},
  title = {{Touché20-Argument-Retrieval-for-Controversial-
Questions}},
  url = {https://doi.org/10.5281/zenodo.6862281},
  year = {2022},
}
""",
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
        adapted_from=["NanoTouche2020Retrieval"],
    )
