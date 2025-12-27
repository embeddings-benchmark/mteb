from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class CanadaTaxCourtOutcomesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CanadaTaxCourtOutcomesLegalBenchClassification",
        description="The input is an excerpt of text from Tax Court of Canada decisions involving appeals of tax related matters. The task is to classify whether the excerpt includes the outcome of the appeal, and if so, to specify whether the appeal was allowed or dismissed. Partial success (e.g. appeal granted on one tax year but dismissed on another) counts as allowed (with the exception of costs orders which are disregarded). Where the excerpt does not clearly articulate an outcome, the system should indicate other as the outcome. Categorizing case outcomes is a common task that legal researchers complete in order to gather datasets involving outcomes in legal processes for the purposes of quantitative empirical legal research.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CanadaTaxCourtOutcomesLegalBenchClassification",
            "revision": "982f04b0361c049d9ac625f4cba536920f6445da",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=["en-CA"],
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
""",
    )


class ContractNLIConfidentialityOfAgreementLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIConfidentialityOfAgreementLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIConfidentialityOfAgreementLegalBenchClassification",
            "revision": "54d44930d700e762c92c2172cdfa9d4d366a3716",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIExplicitIdentificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLIExplicitIdentificationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIExplicitIdentificationLegalBenchClassification",
            "revision": "ff3a6a6ac868585bb238b2119848d91a26a0848e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that Confidential Information may include verbally conveyed information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
            "revision": "067b81db2f83004896c55b41838dd18f0587a194",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLILimitedUseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLILimitedUseLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLILimitedUseLegalBenchClassification",
            "revision": "e4e5d6c15c185ae5445d475266be21dead4d469d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLINoLicensingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLINoLicensingLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLINoLicensingLegalBenchClassification",
            "revision": "9c4990539f763e0c258f4ccd99fd4d274dc7d14c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLINoticeOnCompelledDisclosureLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
            "revision": "1484bb0c5c8ba45bf503fdbdbf27c32acb098f30",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
            "revision": "516094f26d2d1c02ccaf37981cf985a8f2270a45",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIPermissibleCopyLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleCopyLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIPermissibleCopyLegalBenchClassification",
            "revision": "7c0b7ae6cf733aab043feac9ad02ecc9225b0fb3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may independently develop information similar to Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
            "revision": "511df8b7cbab7541d33ccb1973e475b74e172854",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
            "revision": "51f026c1cafb1d4d7f7554416753ff1622c09cb4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLIReturnOfConfidentialInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
            "revision": "472f5049da3c1408ed7a5cba7e0162578ff44efc",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLISharingWithEmployeesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithEmployeesLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLISharingWithEmployeesLegalBenchClassification",
            "revision": "4e2b196d4fa8e21dc93e1e4a6cae1d9beff4e822",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLISharingWithThirdPartiesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithThirdPartiesLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLISharingWithThirdPartiesLegalBenchClassification",
            "revision": "e77c208d1e9f3f0c0524a26450f65b6f772d658a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class ContractNLISurvivalOfObligationsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISurvivalOfObligationsLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that some obligations of Agreement may survive termination of Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/ContractNLISurvivalOfObligationsLegalBenchClassification",
            "revision": "475da8718d284169f082582b2fd707a6f713c296",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1996-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
    )


class CorporateLobbyingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CorporateLobbyingLegalBenchClassification",
        description="The Corporate Lobbying task consists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CorporateLobbyingLegalBenchClassification",
            "revision": "6b6e5f99575322c653b06b684093f3e2ebb616ee",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADAffiliateLicenseLicenseeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicenseeLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if a clause describes a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADAffiliateLicenseLicenseeLegalBenchClassification",
            "revision": "9796ef97a7d8b1b57f629cd0e226578ced8cd8a2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADAffiliateLicenseLicensorLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicensorLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause describes a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADAffiliateLicenseLicensorLegalBenchClassification",
            "revision": "75030cebe1cdcaa83006e5dbde8a240562136b4a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADAntiAssignmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAntiAssignmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires consent or notice of a party if the contract is assigned to a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADAntiAssignmentLegalBenchClassification",
            "revision": "1a733caea65b176d295c6fbbe299db2c25a9bd66",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADAuditRightsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAuditRightsLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause gives a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADAuditRightsLegalBenchClassification",
            "revision": "d3d4840d7dbe198cd8686e6a8c6ebb3d82a5f18d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADCapOnLiabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADCapOnLiabilityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a cap on liability upon the breach of a party's obligation. This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADCapOnLiabilityLegalBenchClassification",
            "revision": "fb58e8c7c1efa60801632a8922933464843e1524",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADChangeOfControlLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADChangeOfControlLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause gives one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADChangeOfControlLegalBenchClassification",
            "revision": "236b760c59f9f210ce7422bb7c0ca49817a18b70",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADCompetitiveRestrictionExceptionLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="CUADCompetitiveRestrictionExceptionLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause mentions exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADCompetitiveRestrictionExceptionLegalBenchClassification",
            "revision": "ef14f6f9a05af36a2c234b6533bf21c0dbcf7cf0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADCovenantNotToSueLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADCovenantNotToSueLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party is restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADCovenantNotToSueLegalBenchClassification",
            "revision": "d6b3577c0b84238c7b0877e7e0c0fee1551d25d5",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADEffectiveDateLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADEffectiveDateLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the agreement becomes effective.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADEffectiveDateLegalBenchClassification",
            "revision": "0cf7e2aca7aa51d486045dca9b3d6b0d07db630c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADExclusivityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADExclusivityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies exclusive dealing commitment with the counterparty. This includes a commitment to procure all 'requirements' from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADExclusivityLegalBenchClassification",
            "revision": "7fe5f86c13d6b9b2d21f8ee2632ccdb7f438624b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADExpirationDateLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADExpirationDateLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the initial term expires.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADExpirationDateLegalBenchClassification",
            "revision": "ef9a5ea18214d19c7a523532901ad41acac9fc1a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADGoverningLawLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADGoverningLawLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies which state/country’s law governs the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADGoverningLawLegalBenchClassification",
            "revision": "4f8f4419104ae22ec6f4550f63567dd85939a04d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADInsuranceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADInsuranceLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if clause creates a requirement for insurance that must be maintained by one party for the benefit of the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADInsuranceLegalBenchClassification",
            "revision": "d1e3806b0211cb14120dfb3aa2d668ef1140fc21",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADIPOwnershipAssignmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADIPOwnershipAssignmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADIPOwnershipAssignmentLegalBenchClassification",
            "revision": "6e5f3569d6dc50b82c97d7f3f2eada093fcd96fd",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADIrrevocableOrPerpetualLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a license grant that is irrevocable or perpetual.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
            "revision": "0f475ade6d931e48217355fce41aab70aa2a5f54",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADJointIPOwnershipLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADJointIPOwnershipLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause provides for joint or shared ownership of intellectual property between the parties to the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADJointIPOwnershipLegalBenchClassification",
            "revision": "297aa6d803123ca25fa7ae8c2c636d6af980ec55",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADLicenseGrantLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADLicenseGrantLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause contains a license granted by one party to its counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADLicenseGrantLegalBenchClassification",
            "revision": "494d24ea85cacfa6d26155f71e10fc295358df7f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADLiquidatedDamagesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADLiquidatedDamagesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause awards either party liquidated damages for breach or a fee upon the termination of a contract (termination fee).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADLiquidatedDamagesLegalBenchClassification",
            "revision": "a4472313099531936edf6da4edbf2152a93391f9",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADMinimumCommitmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADMinimumCommitmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a minimum order size or minimum amount or units per time period that one party must buy from the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADMinimumCommitmentLegalBenchClassification",
            "revision": "622b7b39ee113265bf81df76211890e57c587664",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADMostFavoredNationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADMostFavoredNationLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADMostFavoredNationLegalBenchClassification",
            "revision": "3b9295af6783cacbb93c11f1a9ca277002725562",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNoSolicitOfCustomersLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoSolicitOfCustomersLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNoSolicitOfCustomersLegalBenchClassification",
            "revision": "4f130c5a4664278a8bf4e7ff09d8d85112b3d9d7",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNoSolicitOfEmployeesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoSolicitOfEmployeesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNoSolicitOfEmployeesLegalBenchClassification",
            "revision": "08d62c29b390c12f68fc77162c3e201aa6a4f47e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNonCompeteLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonCompeteLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNonCompeteLegalBenchClassification",
            "revision": "06716101ddbfcca8de4b81445d1c7e7d9e2e0985",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNonDisparagementLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonDisparagementLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires a party not to disparage the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNonDisparagementLegalBenchClassification",
            "revision": "e9339e9fa3f9d55242c70b8576b6ddf2c59f1c8a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNonTransferableLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonTransferableLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause limits the ability of a party to transfer the license being granted to a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNonTransferableLicenseLegalBenchClassification",
            "revision": "05c35442e30bb9b108242be5f39d5208ac5da8ae",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADNoticePeriodToTerminateRenewalLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a notice period required to terminate renewal.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
            "revision": "d81862dca79319283b2c3a6859264e6ee5a9f82f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADPostTerminationServicesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADPostTerminationServicesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause subjects a party to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADPostTerminationServicesLegalBenchClassification",
            "revision": "4f9b9c3038a38352bd4898bfc3af9f2a5b590474",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADPriceRestrictionsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADPriceRestrictionsLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause places a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADPriceRestrictionsLegalBenchClassification",
            "revision": "d5d5b1f3c9cefdf25bec4596bde3bc8c1ee0d996",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADRenewalTermLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRenewalTermLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a renewal term.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADRenewalTermLegalBenchClassification",
            "revision": "0c3f49532f7749b4409c8226832235d4d4ff33cb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADRevenueProfitSharingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRevenueProfitSharingLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause require a party to share revenue or profit with the counterparty for any technology, goods, or services.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADRevenueProfitSharingLegalBenchClassification",
            "revision": "dc46d756261c0c8ebf9f0a27eb9eef6e831e3d68",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADRofrRofoRofnLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRofrRofoRofnLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause grant one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADRofrRofoRofnLegalBenchClassification",
            "revision": "7d876f343eda5de0a63b887f5d10592dddb3ec3e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADSourceCodeEscrowLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADSourceCodeEscrowLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires one party to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADSourceCodeEscrowLegalBenchClassification",
            "revision": "6123e5d008970f92c2bdb14bd075bd2a49de401f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADTerminationForConvenienceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADTerminationForConvenienceLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that one party can terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADTerminationForConvenienceLegalBenchClassification",
            "revision": "d9092f81fe5733a1c6440165e1bbbe9a62e39863",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADThirdPartyBeneficiaryLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADThirdPartyBeneficiaryLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that that there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADThirdPartyBeneficiaryLegalBenchClassification",
            "revision": "bda329e24fb7ea3c723351ecc29b47c752d92bae",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADUncappedLiabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADUncappedLiabilityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party's liability is uncapped upon the breach of its obligation in the contract. This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADUncappedLiabilityLegalBenchClassification",
            "revision": "a2acc56090c3c0ec974b60a811431dc2f75f1f71",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause grants one party an “enterprise,” “all you can eat” or unlimited usage license.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
            "revision": "d96149b2a357664ca9db9697435e4d5887d22346",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADVolumeRestrictionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADVolumeRestrictionLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADVolumeRestrictionLegalBenchClassification",
            "revision": "035541f442c534a44e7540397e1371f25da725e8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class CUADWarrantyDurationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADWarrantyDurationLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a duration of any warranty against defects or errors in technology, products, or services provided under the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/CUADWarrantyDurationLegalBenchClassification",
            "revision": "73aa92a88fa41628e008b9f2f29d5a0360905e35",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class DefinitionClassificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DefinitionClassificationLegalBenchClassification",
        description="This task consists of determining whether or not a sentence from a Supreme Court opinion offers a definition of a term.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/DefinitionClassificationLegalBenchClassification",
            "revision": "fb71cdc96e0cc9e6520e17b6e392f2342e70336d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity1LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity1LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 1).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity1LegalBenchClassification",
            "revision": "683061f975a5f05d021ff507e4994c6d0701c9c9",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity2LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity2LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 2).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity2LegalBenchClassification",
            "revision": "81ac70caea659eec24741fa608e489a9c6948038",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity3LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity3LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 3).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity3LegalBenchClassification",
            "revision": "cf6f9a41c49ee989c7a824717b9e9e51e3afeff5",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity4LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity4LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 4).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity4LegalBenchClassification",
            "revision": "09d27b70ea8260a5be085f986a18413533cb89a4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity5LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity5LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 5).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity5LegalBenchClassification",
            "revision": "dbe9e1015bb672a7f575a8ac6267556dee05dea3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class Diversity6LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity6LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 6).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/Diversity6LegalBenchClassification",
            "revision": "eed8b8fe78647ed2f75cdf95e218cb9f8fe2e2cf",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class FunctionOfDecisionSectionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FunctionOfDecisionSectionLegalBenchClassification",
        description="The task is to classify a paragraph extracted from a written court decision into one of seven possible categories: 1. Facts - The paragraph describes the faction background that led up to the present lawsuit. 2. Procedural History - The paragraph describes the course of litigation that led to the current proceeding before the court. 3. Issue - The paragraph describes the legal or factual issue that must be resolved by the court. 4. Rule - The paragraph describes a rule of law relevant to resolving the issue. 5. Analysis - The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute. 6. Conclusion - The paragraph presents a conclusion of the court. 7. Decree - The paragraph constitutes a decree resolving the dispute.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/FunctionOfDecisionSectionLegalBenchClassification",
            "revision": "b1f5748902f4b6fab7b8cdaaf58d5bb672c55dc3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class InsurancePolicyInterpretationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="InsurancePolicyInterpretationLegalBenchClassification",
        description="Given an insurance claim and policy, determine whether the claim is covered by the policy.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/InsurancePolicyInterpretationLegalBenchClassification",
            "revision": "4bf67103b496d7606c30311e7d9b95f4ade3198b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class InternationalCitizenshipQuestionsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="InternationalCitizenshipQuestionsLegalBenchClassification",
        description="Answer questions about citizenship law from across the world. Dataset was made using the GLOBALCIT citizenship law dataset, by constructing questions about citizenship law as Yes or No questions.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/InternationalCitizenshipQuestionsLegalBenchClassification",
            "revision": "0a353ea3deab3abd54b61f830dbbf4b3e876d346",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1960-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@misc{vink2023globalcit,
  author = {Vink, Maarten and van der Baaren, Luuk and Bauböck, Rainer and Džankić, Jelena and Honohan, Iseult and Manby, Bronwen},
  howpublished = {https://hdl.handle.net/1814/73190},
  publisher = {Global Citizenship Observatory},
  title = {GLOBALCIT Citizenship Law Dataset, v2.0, Country-Year-Mode Data (Acquisition)},
  year = {2023},
}
""",
    )


class JCrewBlockerLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JCrewBlockerLegalBenchClassification",
        description="The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of determining whether the J.Crew Blocker is present in the document.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/JCrewBlockerLegalBenchClassification",
            "revision": "1e23572774338d056497adb86b796f702ee73bdd",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        superseded_by="JCrewBlockerLegalBenchClassification.v2",
    )


class JCrewBlockerLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JCrewBlockerLegalBenchClassification.v2",
        description="The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of determining whether the J.Crew Blocker is present in the document. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/j_crew_blocker_legal_bench",
            "revision": "692cc80266711eaa41d03c9fb168bff60807ee8a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        adapted_from=["JCrewBlockerLegalBenchClassification"],
    )


class LearnedHandsBenefitsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsBenefitsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal post discusses public benefits and social services that people can get from the government, like for food, disability, old age, housing, medical help, unemployment, child care, or other social needs.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsBenefitsLegalBenchClassification",
            "revision": "017c36f4a02f32307645446849975c6f07d2a0bb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsBusinessLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsBusinessLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal question discusses issues faced by people who run small businesses or nonprofits, including around incorporation, licenses, taxes, regulations, and other concerns. It also includes options when there are disasters, bankruptcies, or other problems.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsBusinessLegalBenchClassification",
            "revision": "55da167976361ce16a353c872f0573ce3a4e53d4",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsConsumerLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsConsumerLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsConsumerLegalBenchClassification",
            "revision": "db7c49405e18b49894a71d2547a0107ad0b6353e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsCourtsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsCourtsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsCourtsLegalBenchClassification",
            "revision": "9325a173eebea5fbba5ca1280e0a08f7b02e804f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsCrimeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsCrimeLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsCrimeLegalBenchClassification",
            "revision": "4fa20264aa3eb19371597c0673a5cda5bd8aac21",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsDivorceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsDivorceLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues around filing for divorce, separation, or annulment, getting spousal support, splitting money and property, and following the court processes.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsDivorceLegalBenchClassification",
            "revision": "c4877a01a2ef9b13eb1f70091eb90a6a45104045",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsDomesticViolenceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsDomesticViolenceLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses dealing with domestic violence and abuse, including getting protective orders, enforcing them, understanding abuse, reporting abuse, and getting resources and status if there is abuse.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsDomesticViolenceLegalBenchClassification",
            "revision": "dddf887bc01691c6496632c44070a3f7fd8eb76f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsEducationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEducationLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues around school, including accommodations for special needs, discrimination, student debt, discipline, and other issues in education.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsEducationLegalBenchClassification",
            "revision": "e86b7c1108777fc86f252ff0a8271464fec89031",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsEmploymentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEmploymentLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsEmploymentLegalBenchClassification",
            "revision": "c76e1338e878e91ae6012dfd8b1cb2e966e87567",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsEstatesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEstatesLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses planning for end-of-life, possible incapacitation, and other special circumstances that would prevent a person from making decisions about their own well-being, finances, and property. This includes issues around wills, powers of attorney, advance directives, trusts, guardianships, conservatorships, and other estate issues that people and families deal with.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsEstatesLegalBenchClassification",
            "revision": "aefd901901040e02d56f6d5dce7d30fe7fc28fec",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsFamilyLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsFamilyLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsFamilyLegalBenchClassification",
            "revision": "20ceae69845f3f6d4a97d3f15d24387f1e305899",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsHealthLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsHealthLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues with accessing health services, paying for medical care, getting public benefits for health care, protecting one's rights in medical settings, and other issues related to health.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsHealthLegalBenchClassification",
            "revision": "434816403482adc419566a2fdedecb5da99deae1",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsHousingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsHousingLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsHousingLegalBenchClassification",
            "revision": "85cc1de68b857ec389766dc2dec16a32461ed315",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsImmigrationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsImmigrationLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses visas, asylum, green cards, citizenship, migrant work and benefits, and other issues faced by people who are not full citizens in the US.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsImmigrationLegalBenchClassification",
            "revision": "82381b78b00ce920457bc3742dc492e39cfca61b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsTortsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsTortsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal question discusses problems that one person has with another person (or animal), like when there is a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsTortsLegalBenchClassification",
            "revision": "71eea90b31c9c04d19dfff8fbb7e9a36b53a9319",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LearnedHandsTrafficLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsTrafficLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal post discusses problems with traffic and parking tickets, fees, driver's licenses, and other issues experienced with the traffic system. It also concerns issues with car accidents and injuries, cars' quality, repairs, purchases, and other contracts.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LearnedHandsTrafficLegalBenchClassification",
            "revision": "9ad5acb8b52644eade6be3bffe1d1f4f5e994f42",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@dataset{learned_hands,
  author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
  note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
  title = {LearnedHands Dataset},
  url = {https://spot.suffolklitlab.org/data/#learnedhands},
  urldate = {2022-05-21},
  year = {2022},
}
""",
    )


class LegalReasoningCausalityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LegalReasoningCausalityLegalBenchClassification",
        description="Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/LegalReasoningCausalityLegalBenchClassification",
            "revision": "50ff9872efffaf71c95e1c0cf7f54a6e7c367bcc",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        superseded_by="LegalReasoningCausalityLegalBenchClassification.v2",
    )


class LegalReasoningCausalityLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LegalReasoningCausalityLegalBenchClassification.v2",
        description="Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/legal_reasoning_causality_legal_bench",
            "revision": "563c52ea5216784b608912e67049226ae8cdf702",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        adapted_from=["LegalReasoningCausalityLegalBenchClassification"],
    )


_MAUD_DATASET_MAP = [
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        # The label "A" has only one example and the label "E" has only two examples, so we drop rows with them
        "filter_cols": ["A", "E"],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        "filter_cols": [],
    },
    {
        # The labels "A" and "D" have only two examples, so we drop rows with them
        "filter_cols": ["A", "D"],
    },
    {
        "filter_cols": [],
    },
]


class MAUDLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MAUDLegalBenchClassification",
        description="This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response. This is a combination of all 34 of the MAUD Legal Bench datasets: 1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options. 2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options. 6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options. 7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options. 8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options. 11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options. 12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options. 13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options. 14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options. 15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options. 16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options. 17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options. 18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options. 19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options. 20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options. 21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options. 23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options. 24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options. 25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options. 26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options. 27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer: what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options. 28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options. 29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options. 31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options. 32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options. 33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options. 34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/MAUDLegalBenchClassification",
            "revision": "bc3a598904d4d5c8a642156c4864b48e6420254d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{wang2023maud,
  author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2301.00876},
  title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
  year = {2023},
}
""",
        superseded_by="MAUDLegalBenchClassification.v2",
    )


class MAUDLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MAUDLegalBenchClassification.v2",
        description="This task was constructed from the MAUD dataset, which consists of over 47,000 labels across 152 merger agreements annotated to identify 92 questions in each agreement used by the 2021 American Bar Association (ABA) Public Target Deal Points Study. Each dataset is formatted as a series of multiple-choice questions, where given a segment of the merger agreement and a Deal Point question, the model is to choose the answer that best characterizes the agreement as response. This is a combination of all 34 of the MAUD Legal Bench datasets: 1. MAUD Ability To Consummate Concept Is Subject To MAE Carveouts: Given an excerpt from a merger agreement and the task is to answer: is the “ability to consummate” concept subject to Material Adverse Effect (MAE) carveouts? amongst the multiple choice options. 2. MAUD Accuracy Of Fundamental Target RWS Bringdown Standard: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 3. MAUD Accuracy Of Target Capitalization RW Outstanding Shares Bringdown Standard Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 4. MAUD Accuracy Of Target General RW Bringdown Timing Answer: Given an excerpt from a merger agreement and the task is to answer: how accurate must the fundamental representations and warranties be according to the bring down provision, amongst the multiple choice options. 5. MAUD Additional Matching Rights Period For Modifications Cor: Given an excerpt from a merger agreement and the task is to answer: how long is the additional matching rights period for modifications in case the board changes its recommendation, amongst the multiple choice options. 6. MAUD Application Of Buyer Consent Requirement Negative Interim Covenant: Given an excerpt from a merger agreement and the task is to answer: what negative covenants does the requirement of Buyer consent apply to, amongst the multiple choice options. 7. MAUD Buyer Consent Requirement Ordinary Course: Given an excerpt from a merger agreement and the task is to answer: in case the Buyer's consent for the acquired company's ordinary business operations is required, are there any limitations on the Buyer's right to condition, withhold, or delay their consent, amongst the multiple choice options. 8. MAUD Change In Law Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in law that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 9. MAUD Changes In GAAP Or Other Accounting Principles Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes in GAAP or other accounting principles that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 10. MAUD COR Permitted In Response To Intervening Event: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted in response to an intervening event, amongst the multiple choice options. 11. MAUD COR Permitted With Board Fiduciary Determination Only: Given an excerpt from a merger agreement and the task is to answer: is Change of Recommendation permitted as long as the board determines that such change is required to fulfill its fiduciary obligations, amongst the multiple choice options. 12. MAUD COR Standard Intervening Event: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in response to an intervening event, amongst the multiple choice options. 13. MAUD COR Standard Superior Offer: Given an excerpt from a merger agreement and the task is to answer: what standard should the board follow when determining whether to change its recommendation in connection with a superior offer, amongst the multiple choice options. 14. MAUD Definition Contains Knowledge Requirement Answer: Given an excerpt from a merger agreement and the task is to answer: what is the knowledge requirement in the definition of “Intervening Event”, amongst the multiple choice options. 15. MAUD Definition Includes Asset Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of asset deals, amongst the multiple choice options. 16. MAUD Definition Includes Stock Deals: Given an excerpt from a merger agreement and the task is to answer: what qualifies as a superior offer in terms of stock deals, amongst the multiple choice options. 17. MAUD Fiduciary Exception Board Determination Standard: Given an excerpt from a merger agreement and the task is to answer: under what circumstances could the Board take actions on a different acquisition proposal notwithstanding the no-shop provision, amongst the multiple choice options. 18. MAUD Fiduciary Exception Board Determination Trigger No Shop: Given an excerpt from a merger agreement and the task is to answer: what type of offer could the Board take actions on notwithstanding the no-shop provision, amongst the multiple choice options. 19. MAUD Financial Point Of View Is The Sole Consideration: Given an excerpt from a merger agreement and the task is to answer: is “financial point of view” the sole consideration when determining whether an offer is superior, amongst the multiple choice options. 20. MAUD FLS MAE Standard: Given an excerpt from a merger agreement and the task is to answer: what is the Forward Looking Standard (FLS) with respect to Material Adverse Effect (MAE), amongst the multiple choice options. 21. MAUD General Economic and Financial Conditions Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do changes caused by general economic and financial conditions that have disproportionate impact qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 22. MAUD Includes Consistent With Past Practice: Given an excerpt from a merger agreement and the task is to answer: does the wording of the Efforts Covenant clause include “consistent with past practice”, amongst the multiple choice options. 23. MAUD Initial Matching Rights Period COR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in case the board changes its recommendation, amongst the multiple choice options. 24. MAUD Initial Matching Rights Period FTR: Given an excerpt from a merger agreement and the task is to answer: how long is the initial matching rights period in connection with the Fiduciary Termination Right (FTR), amongst the multiple choice options. 25. MAUDInterveningEventRequiredToOccurAfterSigningAnswer: Given an excerpt from a merger agreement and the task is to answer: is an “Intervening Event” required to occur after signing, amongst the multiple choice options. 26. MAUD Knowledge Definition: Given an excerpt from a merger agreement and the task is to answer: what counts as Knowledge, amongst the multiple choice options. 27. MAUDLiabilityStandardForNoShopBreachByTargetNonDORepresentatives: Given an excerpt from a merger agreement and the task is to answer: what is the liability standard for no-shop breach by Target Non-D&O Representatives, amongst the multiple choice options. 28. MAUD Ordinary Course Efforts Standard: Given an excerpt from a merger agreement and the task is to answer: what is the efforts standard, amongst the multiple choice options. 29. MAUD Pandemic Or Other Public Health Event Subject To Disproportionate Impact Modifier: Given an excerpt from a merger agreement and the task is to answer: do pandemics or other public health events have to have disproportionate impact to qualify for Material Adverse Effect (MAE), amongst the multiple choice options. 30. MAUD Pandemic Or Other Public Health Event Specific Reference To Pandemic Related Governmental Responses Or Measures: Given an excerpt from a merger agreement and the task is to answer: is there specific reference to pandemic-related governmental responses or measures in the clause that qualifies pandemics or other public health events for Material Adverse Effect (MAE), amongst the multiple choice options. 31. MAUD Relational Language MAE Applies To: Given an excerpt from a merger agreement and the task is to answer: what carveouts pertaining to Material Adverse Effect (MAE) does the relational language apply to?, amongst the multiple choice options. 32. MAUD Specific Performance: Given an excerpt from a merger agreement and the task is to answer: what is the wording of the Specific Performance clause regarding the parties' entitlement in the event of a contractual breach, amongst the multiple choice options. 33. MAUD Tail Period Length: Given an excerpt from a merger agreement and the task is to answer: how long is the Tail Period, amongst the multiple choice options. 34. MAUD Type Of Consideration: Given an excerpt from a merger agreement and the task is to answer: what type of consideration is specified in this agreement, amongst the multiple choice options. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/maud_legal_bench",
            "revision": "655744e3745703e6f551e78b4c4cba1702774ce3",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{wang2023maud,
  author = {Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2301.00876},
  title = {MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
  year = {2023},
}
""",
        adapted_from=["MAUDLegalBenchClassification"],
    )


class NYSJudicialEthicsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="NYSJudicialEthicsLegalBenchClassification",
        description="Answer questions on judicial ethics from the New York State Unified Court System Advisory Committee.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/NYSJudicialEthicsLegalBenchClassification",
            "revision": "e5da2c5ca417e3f360afa914fe9aa06094a72325",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class OPP115DataRetentionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115DataRetentionLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes how long user information is stored.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115DataRetentionLegalBenchClassification",
            "revision": "b6600dce343ecbaf9adbc246218ab2a321db9d08",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115DataSecurityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115DataSecurityLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes how user information is protected.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115DataSecurityLegalBenchClassification",
            "revision": "017dc6cf1740eb16a38efc1abde71f88b374a795",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        superseded_by="OPP115DataSecurityLegalBenchClassification.v2",
    )


class OPP115DataSecurityLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115DataSecurityLegalBenchClassification.v2",
        description="Given a clause from a privacy policy, classify if the clause describes how user information is protected. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/opp115_data_security_legal_bench",
            "revision": "8596086d90fa4f2574b15d96a60cb6bc9889806b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        adapted_from=["OPP115DataSecurityLegalBenchClassification"],
    )


class OPP115DoNotTrackLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115DoNotTrackLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115DoNotTrackLegalBenchClassification",
            "revision": "2d2b6b92f2a4c59db5a80be1be8bb3f784b18b42",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        superseded_by="OPP115DoNotTrackLegalBenchClassification.v2",
    )


class OPP115DoNotTrackLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115DoNotTrackLegalBenchClassification.v2",
        description="Given a clause from a privacy policy, classify if the clause describes if and how Do Not Track signals for online tracking and advertising are honored. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/opp115_do_not_track_legal_bench",
            "revision": "3e2cc83cd3fc98dc6d76825c21ed4fbed86d560c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        adapted_from=["OPP115DoNotTrackLegalBenchClassification"],
    )


class OPP115FirstPartyCollectionUseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115FirstPartyCollectionUseLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes how and why a service provider collects user information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115FirstPartyCollectionUseLegalBenchClassification",
            "revision": "fc48a84783f2dbb8d0fc48ab51cbd78c8cb69b5b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115InternationalAndSpecificAudiencesLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="OPP115InternationalAndSpecificAudiencesLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describe practices that pertain only to a specific group of users (e.g., children, Europeans, or California residents).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115InternationalAndSpecificAudiencesLegalBenchClassification",
            "revision": "6a21c785613c682f4e756e5d6802b3677891c9cf",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115PolicyChangeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115PolicyChangeLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes if and how users will be informed about changes to the privacy policy.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115PolicyChangeLegalBenchClassification",
            "revision": "614c5f149381d163cb3027dad8c6962bb9c2240e",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115ThirdPartySharingCollectionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115ThirdPartySharingCollectionLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describe how user information may be shared with or collected by third parties.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115ThirdPartySharingCollectionLegalBenchClassification",
            "revision": "f54384d5c945394d86d0c9465b93d466c2119f1b",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115UserAccessEditAndDeletionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115UserAccessEditAndDeletionLegalBenchClassification",
        description="Given a clause from a privacy policy, classify if the clause describes if and how users may access, edit, or delete their information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115UserAccessEditAndDeletionLegalBenchClassification",
            "revision": "e78f879576b91b23d0b4d8cf2ba95246c7041d72",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
    )


class OPP115UserChoiceControlLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115UserChoiceControlLegalBenchClassification",
        description="Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OPP115UserChoiceControlLegalBenchClassification",
            "revision": "9079921505df03be356c71ba2e62a3445de18d87",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        superseded_by="OPP115UserChoiceControlLegalBenchClassification.v2",
    )


class OPP115UserChoiceControlLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OPP115UserChoiceControlLegalBenchClassification.v2",
        description="Given a clause fro ma privacy policy, classify if the clause describes the choices and control options available to users. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/opp115_user_choice_control_legal_bench",
            "revision": "f308b16f8baee2080cf43e28ff01d93032d51eee",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{wilson2016creation,
  author = {Wilson, Shomir and Schaub, Florian and Dara, Aswarth Abhilash and Liu, Frederick and Cherivirala, Sushain and Leon, Pedro Giovanni and Andersen, Mads Schaarup and Zimmeck, Sebastian and Sathyendra, Kanthashree Mysore and Russell, N Cameron and others},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages = {1330--1340},
  title = {The creation and analysis of a website privacy policy corpus},
  year = {2016},
}
""",
        adapted_from=["OPP115UserChoiceControlLegalBenchClassification"],
    )


class OralArgumentQuestionPurposeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OralArgumentQuestionPurposeLegalBenchClassification",
        description="This task classifies questions asked by Supreme Court justices at oral argument into seven categories: 1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing 2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for 3. Implications - questions about the limits of a rule or its implications for future cases 4. Support - questions offering support for the advocate’s position 5. Criticism - questions criticizing an advocate’s position 6. Communicate - question designed primarily to communicate with other justices 7. Humor - questions designed to interject humor into the argument and relieve tension",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OralArgumentQuestionPurposeLegalBenchClassification",
            "revision": "6e2014c2258f123193ee5c1c4955777d2f02dcac",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        superseded_by="OralArgumentQuestionPurposeLegalBenchClassification.v2",
    )


class OralArgumentQuestionPurposeLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OralArgumentQuestionPurposeLegalBenchClassification.v2",
        description="This task classifies questions asked by Supreme Court justices at oral argument into seven categories: 1. Background - questions seeking factual or procedural information that is missing or not clear in the briefing 2. Clarification - questions seeking to get an advocate to clarify her position or the scope of the rule being advocated for 3. Implications - questions about the limits of a rule or its implications for future cases 4. Support - questions offering support for the advocate’s position 5. Criticism - questions criticizing an advocate’s position 6. Communicate - question designed primarily to communicate with other justices 7. Humor - questions designed to interject humor into the argument and relieve tension This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/oral_argument_question_purpose_legal_bench",
            "revision": "cdc020e244cb846ce4e0325cb602cf04126c79d2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2021-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
        adapted_from=["OralArgumentQuestionPurposeLegalBenchClassification"],
    )


class OverrulingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OverrulingLegalBenchClassification",
        description="This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/OverrulingLegalBenchClassification",
            "revision": "9709f47804a54d1831efeea26989997ae4d04525",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1965-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{zheng2021does,
  author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
  booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
  pages = {159--168},
  title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
  year = {2021},
}
""",
        superseded_by="OverrulingLegalBenchClassification.v2",
    )


class OverrulingLegalBenchClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OverrulingLegalBenchClassification.v2",
        description="This task consists of classifying whether or not a particular sentence of case law overturns the decision of a previous case. This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/overruling_legal_bench",
            "revision": "fee708d1959b3258bc3e408afdd3e6c2051adf80",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1965-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@inproceedings{zheng2021does,
  author = {Zheng, Lucia and Guha, Neel and Anderson, Brandon R and Henderson, Peter and Ho, Daniel E},
  booktitle = {Proceedings of the eighteenth international conference on artificial intelligence and law},
  pages = {159--168},
  title = {When does pretraining help? assessing self-supervised learning for law and the casehold dataset of 53,000+ legal holdings},
  year = {2021},
}
""",
        adapted_from=["OverrulingLegalBenchClassification"],
    )


class PersonalJurisdictionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersonalJurisdictionLegalBenchClassification",
        description="Given a fact pattern describing the set of contacts between a plaintiff, defendant, and forum, determine if a court in that forum could exercise personal jurisdiction over the defendant.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/PersonalJurisdictionLegalBenchClassification",
            "revision": "0ffe9bd33f7358d53db66b0252eebb06424d6565",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class PROALegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PROALegalBenchClassification",
        description="Given a statute, determine if the text contains an explicit private right of action. Given a privacy policy clause and a description of the clause, determine if the description is correct. A private right of action (PROA) exists when a statute empowers an ordinary individual (i.e., a private person) to legally enforce their rights by bringing an action in court. In short, a PROA creates the ability for an individual to sue someone in order to recover damages or halt some offending conduct. PROAs are ubiquitous in antitrust law (in which individuals harmed by anti-competitive behavior can sue offending firms for compensation) and environmental law (in which individuals can sue entities which release hazardous substances for damages).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/PROALegalBenchClassification",
            "revision": "b5ebcae6d49ea4484fd8b8e5c744f2718da152d0",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class SCDBPAccountabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPAccountabilityLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer maintains internal compliance procedures on company standards regarding human trafficking and slavery? This includes any type of internal accountability mechanism. Requiring independently of the supply to comply with laws does not qualify or asking for documentary evidence of compliance does not count either.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDBPAccountabilityLegalBenchClassification",
            "revision": "42eeb5db89e78e0bda5b782036eaf9933aa51a24",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDBPAuditsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPAuditsLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDBPAuditsLegalBenchClassification",
            "revision": "601d460774fef3752e3488836c358e5bd0283044",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDBPCertificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPCertificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDBPCertificationLegalBenchClassification",
            "revision": "6d3296bfd8d16d0eb61e320b1dfccbd67ca6b9bb",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDBPTrainingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPTrainingLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  provides training to employees on human trafficking and slavery? Broad policies such as ongoing dialogue on mitigating risks of human trafficking and slavery or increasing managers and purchasers knowledge about health, safety and labor practices qualify as training. Providing training to contractors who failed to comply with human trafficking laws counts as training.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDBPTrainingLegalBenchClassification",
            "revision": "9a2703cfd9f77d7b5871773b4bc1dd2a94e7999c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDBPVerificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPVerificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer engages in verification and auditing as one practice, expresses that it may conduct an audit, or expressess that it is assessing supplier risks through a review of the US Dept. of Labor's List?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDBPVerificationLegalBenchClassification",
            "revision": "ed8e5b77e4b00a4b2c836a48e817ccbd68bd66a6",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDDAccountabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDDAccountabilityLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer maintains internal accountability standards and procedures for employees or contractors failing to meet company standards regarding slavery and trafficking?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDDAccountabilityLegalBenchClassification",
            "revision": "be30c634c1261d0ec0650302e27e79f412f75cae",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDDAuditsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDDAuditsLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer conducts audits of suppliers to evaluate supplier compliance with company standards for trafficking and slavery in supply chains? The disclosure shall specify if the verification was not an independent, unannounced audit.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDDAuditsLegalBenchClassification",
            "revision": "c267c0134f6d7c7ea4488e91be465b46dc97858d",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDDCertificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDDCertificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer requires direct suppliers to certify that materials incorporated into the product comply with the laws regarding slavery and human trafficking of the country or countries in which they are doing business?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDDCertificationLegalBenchClassification",
            "revision": "c6bc21587ed835f528afcd8476357a97ebb63dd7",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDDTrainingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDDTrainingLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer provides company employees and management, who have direct responsibility for supply chain management, training on human trafficking and slavery, particularly with respect to mitigating risks within the supply chains of products?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDDTrainingLegalBenchClassification",
            "revision": "75cb38ca7c0cf5bcdbed96e48ad62eb72d3ec9d2",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class SCDDVerificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDDVerificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose to what extent, if any, that the retail seller or manufacturer engages in verification of product supply chains to evaluate and address risks of human trafficking and slavery? If the company conducts verification], the disclosure shall specify if the verification was not conducted by a third party.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/SCDDVerificationLegalBenchClassification",
            "revision": "2522b3b3d50d892f8e40193202e26f328a383c1a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{chilton2017limitations,
  author = {Chilton, Adam S and Sarfaty, Galit A},
  journal = {Stan. J. Int'l L.},
  pages = {1},
  publisher = {HeinOnline},
  title = {The limitations of supply chain disclosure regimes},
  volume = {53},
  year = {2017},
}

@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
    )


class TelemarketingSalesRuleLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TelemarketingSalesRuleLegalBenchClassification",
        description="Determine how 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) (governing deceptive practices) apply to different fact patterns. This dataset is designed to test a model’s ability to apply 16 C.F.R. § 310.3(a)(1) and 16 C.F.R. § 310.3(a)(2) of the Telemarketing Sales Rule to a simple fact pattern with a clear outcome. Each fact pattern ends with the question: “Is this a violation of the Telemarketing Sales Rule?” Each fact pattern is paired with the answer “Yes” or the answer “No.” Fact patterns are listed in the column “text,” and answers are listed in the column “label.”",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/TelemarketingSalesRuleLegalBenchClassification",
            "revision": "84366cfb3682a3f900039cbc45f7672c119f662c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class TextualismToolDictionariesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TextualismToolDictionariesLegalBenchClassification",
        description="Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the dictionary meaning of terms.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/TextualismToolDictionariesLegalBenchClassification",
            "revision": "b80f65972c09c36e5e663aa0b8a675f8391ab317",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class TextualismToolPlainLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TextualismToolPlainLegalBenchClassification",
        description="Determine if a paragraph from a judicial opinion is applying a form textualism that relies on the ordinary (“plain”) meaning of terms.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/TextualismToolPlainLegalBenchClassification",
            "revision": "aaf4454776f64c37f0e6cda45ea3af43f6266b47",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class UCCVCommonLawLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UCCVCommonLawLegalBenchClassification",
        description="Determine if a contract is governed by the Uniform Commercial Code (UCC) or the common law of contracts.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/UCCVCommonLawLegalBenchClassification",
            "revision": "71bbb6b040c7a4477a3e4d9a820a063462e9fb01",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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
""",
    )


class UnfairTOSLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="UnfairTOSLegalBenchClassification",
        description="Given a clause from a terms-of-service contract, determine the category the clause belongs to. The purpose of this task is classifying clauses in Terms of Service agreements. Clauses have been annotated by into nine categories: ['Arbitration', 'Unilateral change', 'Content removal', 'Jurisdiction', 'Choice of law', 'Limitation of liability', 'Unilateral termination', 'Contract by using', 'Other']. The first eight categories correspond to clauses that would potentially be deemed potentially unfair. The last category (Other) corresponds to clauses in agreements which don’t fit into these categories.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/UnfairTOSLegalBenchClassification",
            "revision": "f543f88c088f01961ee9b5a6a1db8cc767693f25",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2006-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
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

@article{lippi2019claudette,
  author = {Lippi, Marco and Pa{\l}ka, Przemys{\l}aw and Contissa, Giuseppe and Lagioia, Francesca and Micklitz, Hans-Wolfgang and Sartor, Giovanni and Torroni, Paolo},
  journal = {Artificial Intelligence and Law},
  pages = {117--139},
  publisher = {Springer},
  title = {CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service},
  volume = {27},
  year = {2019},
}
""",
    )
