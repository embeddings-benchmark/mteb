from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LegalBenchCorporateLobbying(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LegalBenchCorporateLobbying",
        description="The dataset includes bill titles and bill summaries related to corporate lobbying.",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying",
        dataset={
            "path": "mteb/legalbench_corporate_lobbying",
            "revision": "f69691c650464e62546d7f2a4536f8f87c891e38",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{guha2023legalbench,
  author = {Guha, Neel and Nyarko, Julian and Ho, Daniel and Ré, Christopher and Chilton, Adam and K, Aditya and Chohlas-Wood, Alex and Peters, Austin and Waldon, Brandon and Rockmore, Daniel and Zambrano, Diego and Talisman, Dmitry and Hoque, Enam and Surani, Faiz and Fagan, Frank and Sarfaty, Galit and Dickinson, Gregory and Porat, Haggai and Hegland, Jason and Wu, Jessica and Nudell, Joe and Niklaus, Joel and Nay, John and Choi, Jonathan and Tobia, Kevin and Hagan, Margaret and Ma, Megan and Livermore, Michael and Rasumov-Rahe, Nikon and Holzenberger, Nils and Kolt, Noam and Henderson, Peter and Rehaag, Sean and Goel, Sharad and Gao, Shang and Williams, Spencer and Gandhi, Sunny and Zur, Tom and Iyer, Varun and Li, Zehua},
  booktitle = {Advances in Neural Information Processing Systems 36},
  collection = {NeurIPS 2023},
  doi = {10.52202/075280-1915},
  pages = {44123–44279},
  publisher = {Neural Information Processing Systems Foundation, Inc. (NeurIPS)},
  series = {NeurIPS 2023},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  url = {https://doi.org/10.52202/075280-1915},
  year = {2023},
}

@inproceedings{hendrycks2021cuad,
  archiveprefix = {arXiv},
  author = {Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  booktitle = {Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  eprint = {2103.06268},
  title = {{CUAD}: An Expert-Annotated {NLP} Dataset for Legal Contract Review},
  url = {https://openreview.net/forum?id=7l1Ygs3Bamw},
  year = {2021},
}

@inproceedings{holzenberger2021factoring,
  address = {Online},
  author = {Holzenberger, Nils and Van Durme, Benjamin},
  booktitle = {Proceedings of the 59th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  doi = {10.18653/v1/2021.acl-long.213},
  editor = {Zong, Chengqing and Xia, Fei and Li, Wenjie and Navigli, Roberto},
  month = {aug},
  pages = {2742--2758},
  publisher = {Association for Computational Linguistics},
  title = {Factoring Statutory Reasoning as Language Understanding Challenges},
  url = {https://aclanthology.org/2021.acl-long.213/},
  year = {2021},
}

@inproceedings{koreeda2021contractnli,
  address = {Punta Cana, Dominican Republic},
  author = {Koreeda, Yuta and Manning, Christopher},
  booktitle = {Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2021},
  doi = {10.18653/v1/2021.findings-emnlp.164},
  editor = {Moens, Marie-Francine and Huang, Xuanjing and Specia, Lucia and Yih, Scott Wen-tau},
  month = {nov},
  pages = {1907--1919},
  publisher = {Association for Computational Linguistics},
  title = {{C}ontract{NLI}: A Dataset for Document-level Natural Language Inference for Contracts},
  url = {https://aclanthology.org/2021.findings-emnlp.164/},
  year = {2021},
}

@article{lippi2019claudette,
  author = {Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
  journal = {Artificial Intelligence and Law},
  pages = {117--139},
  publisher = {Springer},
  title = {CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
  volume = {27},
  year = {2019},
}

@inproceedings{ravichander2019question,
  author = {Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  doi = {10.18653/v1/d19-1500},
  pages = {4946–4957},
  publisher = {Association for Computational Linguistics},
  title = {Question Answering for Privacy Policies: Combining Computational and Legal Perspectives},
  url = {https://doi.org/10.18653/v1/d19-1500},
  year = {2019},
}

@inproceedings{wang2023maud,
  address = {Singapore},
  author = {Wang, Steven and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dmitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2023.emnlp-main.1019},
  editor = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  month = {dec},
  pages = {16369--16382},
  publisher = {Association for Computational Linguistics},
  title = {{MAUD}: An Expert-Annotated Legal {NLP} Dataset for Merger Agreement Understanding},
  url = {https://aclanthology.org/2023.emnlp-main.1019/},
  year = {2023},
}

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}

@inproceedings{zheng2021does,
  author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
  booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
  pages = {159--168},
  title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
  year = {2021},
}

@article{zimmeck2019maps,
  author = {Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
  journal = {Proc. Priv. Enhancing Tech.},
  pages = {66},
  title = {Maps: Scaling privacy compliance analysis to a million apps},
  volume = {2019},
  year = {2019},
}
""",
    )
