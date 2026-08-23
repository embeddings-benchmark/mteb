from mteb.abstasks.retrieval import AbsTaskRetrieval
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


class MultilingualNanoArguAnaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoArguAnaRetrieval",
        description="A small translated argument retrieval dataset for debate contexts. The task is to retrieve the best counter-argument for a given argument. It consists of arguments sourced from debate portals on various controversial topics. This is a machine-translated multilingual subset of the ArguAna dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoArguAnaRetrieval",
            "revision": "c0b2a891577e87141baf768b28e7acb997ac5680",
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


class MultilingualNanoClimateFeverRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoClimateFeverRetrieval",
        description="A small translated fact-checking dataset focused on climate change. The task is to retrieve Wikipedia articles that support or refute a given real-world claim regarding climate change, adopting the FEVER methodology. This is a machine-translated multilingual subset of the ClimateFEVER dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoClimateFeverRetrieval",
            "revision": "2fe8eb61630d248dc0bd1a13e4b7f6406b4a3e2a",
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


class MultilingualNanoDBPediaRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoDBPediaRetrieval",
        description="A small translated entity retrieval dataset. The task is to retrieve relevant entity descriptions from the DBpedia knowledge base in response to structured or unstructured natural language search queries. This is a machine-translated multilingual subset of the DBpedia-Entity v2 dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoDBPediaRetrieval",
            "revision": "d190b64f4888205f70b5bfdd4fdf806100139dd0",
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


class MultilingualNanoFEVERRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoFEVERRetrieval",
        description="A small translated fact-checking and claim verification dataset. The task is to retrieve relevant Wikipedia articles that either support or refute a given claim. This is a machine-translated multilingual subset of the FEVER dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoFEVERRetrieval",
            "revision": "58c0c39b5718de4c64ba115e30ae830f25fc6c62",
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


class MultilingualNanoFiQA2018Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoFiQA2018Retrieval",
        description="A small translated question-answering dataset focused on the financial domain. The task is to retrieve relevant forum replies or posts that answer a given financial query. This is a machine-translated multilingual subset of the FiQA-2018 dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoFiQA2018Retrieval",
            "revision": "6af7d1d165dab5c41840440ee33373d3a459483a",
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


class MultilingualNanoHotpotQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoHotpotQARetrieval",
        description="A small translated multi-hop question answering dataset. Sourced from Wikipedia, the task is to retrieve documents containing the supporting evidence needed to answer complex, multi-hop questions. This is a machine-translated multilingual subset of the HotpotQA dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoHotpotQARetrieval",
            "revision": "d226781cb805c360229d14c2efa8672d11226075",
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


class MultilingualNanoMSMARCORetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoMSMARCORetrieval",
        description="A small translated passage retrieval dataset based on real-world search queries. The task is to retrieve relevant web passages that answer a given search query submitted to the Bing search engine. This is a machine-translated multilingual subset of the MS MARCO dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoMSMARCORetrieval",
            "revision": "f579311c703483afcec3062f58fe69e9f0813843",
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


class MultilingualNanoNFCorpusRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoNFCorpusRetrieval",
        description="A small translated medical information retrieval dataset. Sourced from NutritionFacts.org, the task is to retrieve full-text medical documents or scientific abstracts that are relevant to natural language medical queries. This is a machine-translated multilingual subset of the NFCorpus dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoNFCorpusRetrieval",
            "revision": "4484b487f178125827f7dda052f1b2f5673548d6",
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


class MultilingualNanoNQRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoNQRetrieval",
        description="A small translated question answering dataset based on real Google search queries. The task is to retrieve Wikipedia passages that contain the answers to questions submitted by real users. This is a machine-translated multilingual subset of the Natural Questions dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoNQRetrieval",
            "revision": "4104e3376fe243f0bd4845e204b01c0fc3f7d1d7",
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


class MultilingualNanoQuoraRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoQuoraRetrieval",
        description="A small translated duplicate question retrieval dataset sourced from the Quora platform. The task is to retrieve duplicate or semantically equivalent questions for a given query question. This is a machine-translated multilingual subset of the Quora dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoQuoraRetrieval",
            "revision": "eb9418faade7449e869be388846f287aa3535b64",
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


class MultilingualNanoSCIDOCSRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoSCIDOCSRetrieval",
        description="A small translated scientific document retrieval dataset. The task is to retrieve relevant paper abstracts that are cited by a given scientific paper title. This is a machine-translated multilingual subset of the SciDocs dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoSCIDOCSRetrieval",
            "revision": "06cff48c8ebea5f46e3effa29573fc22a5b9d8fc",
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


class MultilingualNanoSciFactRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoSciFactRetrieval",
        description="A small translated scientific claim verification dataset. The task is to retrieve evidence or abstract documents from research literature that support or refute a given scientific claim. This is a machine-translated multilingual subset of the SciFact dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoSciFactRetrieval",
            "revision": "3e75543b8006c013ec14d36a17dd3010eb139f72",
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


class MultilingualNanoTouche2020Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MultilingualNanoTouche2020Retrieval",
        description="A small translated argument retrieval dataset where the task is to retrieve relevant arguments for controversial, debate-style questions. It contains 49 topics phrased as questions on socially debated issues, with a corpus of arguments drawn from the args.me portal. This is a machine-translated multilingual subset of the Touché2020 dataset.",
        reference="https://huggingface.co/datasets/LiquidAI/nanobeir-multilingual-extended",
        dataset={
            "path": "mteb/MultilingualNanoTouche2020Retrieval",
            "revision": "cc244b553d65a88fb66d913907146fe4c33aef29",
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
