from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


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
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}

@article{hendrycks2021cuad,
  author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {arXiv preprint arXiv:2103.06268},
  title = {Cuad: An expert-annotated nlp dataset for legal contract review},
  year = {2021},
}

@article{holzenberger2021factoring,
  author = {Holzenberger, Nils and Van Durme, Benjamin},
  journal = {arXiv preprint arXiv:2105.07903},
  title = {Factoring statutory reasoning as language understanding challenges},
  year = {2021},
}

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
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

@article{ravichander2019question,
  author = {Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
  journal = {arXiv preprint arXiv:1911.00841},
  title = {Question answering for privacy policies: Combining computational and legal perspectives},
  year = {2019},
}

@article{wang2023maud,
  author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2301.00876},
  title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
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
