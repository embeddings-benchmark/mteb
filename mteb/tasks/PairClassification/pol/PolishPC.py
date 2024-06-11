from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SickePLPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SICK-E-PL",
        dataset={
            "path": "PL-MTEB/sicke-pl-pairclassification",
            "revision": "71bba34b0ece6c56dfcf46d9758a27f7a90f17e9",
        },
        description="Polish version of SICK dataset for textual entailment.",
        reference="https://aclanthology.org/2020.lrec-1.207",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{dadas-etal-2020-evaluation,
    title = "Evaluation of Sentence Representations in {P}olish",
    author = "Dadas, Slawomir  and
      Pere{\l}kiewicz, Micha{\l}  and
      Po{\'s}wiata, Rafa{\l}",
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
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.207",
    pages = "1674--1680",
    abstract = "Methods for learning sentence representations have been actively developed in recent years. However, the lack of pre-trained models and datasets annotated at the sentence level has been a problem for low-resource languages such as Polish which led to less interest in applying these methods to language-specific tasks. In this study, we introduce two new Polish datasets for evaluating sentence embeddings and provide a comprehensive evaluation of eight sentence representation methods including Polish and multilingual models. We consider classic word embedding models, recently developed contextual embeddings and multilingual sentence encoders, showing strengths and weaknesses of specific approaches. We also examine different methods of aggregating word vectors into a single sentence vector.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}""",
        n_samples=None,
        avg_character_length=None,
    )


class PpcPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PpcPC",
        dataset={
            "path": "PL-MTEB/ppc-pairclassification",
            "revision": "2c7d2df57801a591f6b1e3aaf042e7a04ec7d9f2",
        },
        description="Polish Paraphrase Corpus",
        reference="https://arxiv.org/pdf/2207.12759.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@misc{dadas2022training,
      title={Training Effective Neural Sentence Encoders from Automatically Mined Paraphrases}, 
      author={Sławomir Dadas},
      year={2022},
      eprint={2207.12759},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}""",
        n_samples=None,
        avg_character_length=None,
    )


class CdscePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="CDSC-E",
        dataset={
            "path": "PL-MTEB/cdsce-pairclassification",
            "revision": "0a3d4aa409b22f80eb22cbf59b492637637b536d",
        },
        description="Compositional Distributional Semantics Corpus for textual entailment.",
        reference="https://aclanthology.org/P17-1073.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{wroblewska-krasnowska-kieras-2017-polish,
    title = "{P}olish evaluation dataset for compositional distributional semantics models",
    author = "Wr{\'o}blewska, Alina  and
      Krasnowska-Kiera{\'s}, Katarzyna",
    editor = "Barzilay, Regina  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1073",
    doi = "10.18653/v1/P17-1073",
    pages = "784--792",
    abstract = "The paper presents a procedure of building an evaluation dataset. for the validation of compositional distributional semantics models estimated for languages other than English. The procedure generally builds on steps designed to assemble the SICK corpus, which contains pairs of English sentences annotated for semantic relatedness and entailment, because we aim at building a comparable dataset. However, the implementation of particular building steps significantly differs from the original SICK design assumptions, which is caused by both lack of necessary extraneous resources for an investigated language and the need for language-specific transformation rules. The designed procedure is verified on Polish, a fusional language with a relatively free word order, and contributes to building a Polish evaluation dataset. The resource consists of 10K sentence pairs which are human-annotated for semantic relatedness and entailment. The dataset may be used for the evaluation of compositional distributional semantics models of Polish.",
}""",
        n_samples=None,
        avg_character_length=None,
    )


class PscPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PSC",
        dataset={
            "path": "PL-MTEB/psc-pairclassification",
            "revision": "d05a294af9e1d3ff2bfb6b714e08a24a6cabc669",
        },
        description="Polish Summaries Corpus",
        reference="http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["pol-Latn"],
        main_score="ap",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation="""@inproceedings{ogrodniczuk-kopec-2014-polish,
    title = "The {P}olish Summaries Corpus",
    author = "Ogrodniczuk, Maciej  and
      Kope{\'c}, Mateusz",
    editor = "Calzolari, Nicoletta  and
      Choukri, Khalid  and
      Declerck, Thierry  and
      Loftsson, Hrafn  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)",
    month = may,
    year = "2014",
    address = "Reykjavik, Iceland",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2014/pdf/1211_Paper.pdf",
    pages = "3712--3715",
    abstract = "This article presents the Polish Summaries Corpus, a new resource created to support the development and evaluation of the tools for automated single-document summarization of Polish. The Corpus contains a large number of manual summaries of news articles, with many independently created summaries for a single text. Such approach is supposed to overcome the annotator bias, which is often described as a problem during the evaluation of the summarization algorithms against a single gold standard. There are several summarizers developed specifically for Polish language, but their in-depth evaluation and comparison was impossible without a large, manually created corpus. We present in detail the process of text selection, annotation process and the contents of the corpus, which includes both abstract free-word summaries, as well as extraction-based summaries created by selecting text spans from the original document. Finally, we describe how that resource could be used not only for the evaluation of the existing summarization tools, but also for studies on the human summarization process in Polish language.",
}""",
        n_samples=None,
        avg_character_length=None,
    )
