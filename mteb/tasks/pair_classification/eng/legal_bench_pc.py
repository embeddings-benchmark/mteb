from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_COLUMN_MAP = [
    {
        "name": "citation_prediction_classification",
        "sent1": "citation",
        "sent2": "text",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "consumer_contracts_qa",
        "sent1": "question",
        "sent2": "contract",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "contract_qa",
        "sent1": "question",
        "sent2": "text",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "hearsay",
        "sent1": "text",
        "sent2": "slice",
        "labels": "answer",
        "mapping": {"yes": 1, "no": 0},
    },
    {
        "name": "privacy_policy_entailment",
        "sent1": "text",
        "sent2": "description",
        "labels": "answer",
        "mapping": {"correct": 1, "incorrect": 0},
    },
    {
        "name": "privacy_policy_qa",
        "sent1": "text",
        "sent2": "question",
        "labels": "answer",
        "mapping": {"relevant": 1, "irrelevant": 0},
    },
]


class LegalBenchPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LegalBenchPC",
        description="This LegalBench pair classification task is a combination of the following datasets: - Citation Prediction Classification: Given a legal statement and a case citation, determine if the citation is supportive of the legal statement. - Consumer Contracts QA: The task consists of 400 yes/no questions relating to consumer contracts (specifically, online terms of service) and is relevant to the legal skill of contract interpretation. - Contract QA: Answer yes/no questions about whether contractual clauses discuss particular issues like confidentiality requirements, BIPA consent, PII data breaches, breach of contract etc. - Hearsay: Classify if a particular piece of evidence qualifies as hearsay. Each sample in the dataset describes (1) an issue being litigated or an assertion a party wishes to prove, and (2) a piece of evidence a party wishes to introduce. The goal is to determine if—as it relates to the issue—the evidence would be considered hearsay under the definition provided above. - Privacy Policy Entailment: Given a privacy policy clause and a description of the clause, determine if the description is correct. This is a binary classification task in which the LLM is provided with a clause from a privacy policy, and a description of that clause (e.g., “The policy describes collection of the user’s HTTP cookies, flash cookies, pixel tags, or similar identifiers by a party to the contract.”). - Privacy Policy QA: Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question. This is a binary classification task in which the LLM is provided with a question (e.g., “do you publish my data”) and a clause from a privacy policy. The LLM must determine if the clause contains an answer to the question, and classify the question-clause pair.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LegalBenchPC",
            "revision": "88408f054cc56b9ec8adc8113ffd65585995c099",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
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

@article{kolt2022predicting,
  author = {Kolt, Noam},
  journal = {Berkeley Tech. LJ},
  pages = {71},
  publisher = {HeinOnline},
  title = {Predicting consumer contracts},
  volume = {37},
  year = {2022},
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
