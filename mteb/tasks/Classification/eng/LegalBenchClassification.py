from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CanadaTaxCourtOutcomesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CanadaTaxCourtOutcomesLegalBenchClassification",
        description="The input is an excerpt of text from Tax Court of Canada decisions involving appeals of tax related matters. The task is to classify whether the excerpt includes the outcome of the appeal, and if so, to specify whether the appeal was allowed or dismissed. Partial success (e.g. appeal granted on one tax year but dismissed on another) counts as allowed (with the exception of costs orders which are disregarded). Where the excerpt does not clearly articulate an outcome, the system should indicate other as the outcome. Categorizing case outcomes is a common task that legal researchers complete in order to gather datasets involving outcomes in legal processes for the purposes of quantitative empirical legal research.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "canada_tax_court_outcomes",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=["en-CA"],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }""",
        n_samples={"test": 244},
        avg_character_length={"test": 622.60},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIConfidentialityOfAgreementLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIConfidentialityOfAgreementLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA provides that the Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_confidentiality_of_agreement",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 82},
        avg_character_length={"test": 473.17},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIExplicitIdentificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLIExplicitIdentificationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that all Confidential Information shall be expressly identified by the Disclosing Party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_explicit_identification",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 109},
        avg_character_length={"test": 506.12},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that Confidential Information may include verbally conveyed information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_inclusion_of_verbally_conveyed_information",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 139},
        avg_character_length={"test": 525.75},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLILimitedUseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLILimitedUseLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_limited_use",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 208},
        avg_character_length={"test": 407.51},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLINoLicensingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLINoLicensingLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Agreement shall not grant Receiving Party any right to Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_no_licensing",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 162},
        avg_character_length={"test": 419.42},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLINoticeOnCompelledDisclosureLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_notice_on_compelled_disclosure",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 142},
        avg_character_length={"test": 503.45},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may acquire information similar to Confidential Information from a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_permissible_acquirement_of_similar_information",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 178},
        avg_character_length={"test": 427.40},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIPermissibleCopyLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleCopyLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may create a copy of some Confidential Information in some circumstances.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_permissible_copy",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 87},
        avg_character_length={"test": 386.84},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may independently develop information similar to Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_permissible_development_of_similar_information",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 136},
        avg_character_length={"test": 396.40},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_permissible_post-agreement_possession",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 111},
        avg_character_length={"test": 529.09},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLIReturnOfConfidentialInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_return_of_confidential_information",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 66},
        avg_character_length={"test": 478.29},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLISharingWithEmployeesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithEmployeesLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_sharing_with_employees",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 170},
        avg_character_length={"test": 548.63},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLISharingWithThirdPartiesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithThirdPartiesLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that the Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_sharing_with_third-parties",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 180},
        avg_character_length={"test": 517.29},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class ContractNLISurvivalOfObligationsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISurvivalOfObligationsLegalBenchClassification",
        description="This task is a subset of ContractNLI, and consists of determining whether a clause from an NDA clause provides that some obligations of Agreement may survive termination of Agreement.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "contract_nli_survival_of_obligations",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{koreeda2021contractnli,
            title={ContractNLI: A dataset for document-level natural language inference for contracts},
            author={Koreeda, Yuta and Manning, Christopher D},
            journal={arXiv preprint arXiv:2110.01799},
            year={2021}
        }""",
        n_samples={"test": 157},
        avg_character_length={"test": 417.64},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CorporateLobbyingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CorporateLobbyingLegalBenchClassification",
        description="The Corporate Lobbying task cosists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "corporate_lobbying",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        """,
        n_samples={"test": 490},
        avg_character_length={"test": 6039.85},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        self.dataset = self.dataset.map(
            lambda examples: {
                "text": examples["bill_title"]
                + "\n\n"
                + examples["bill_summary"]
                + "\n\n"
                + examples["company_name"]
                + "\n\n"
                + examples["company_description"]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["bill_title", "bill_summary", "company_name", "company_description"]
        )


class CUADAffiliateLicenseLicenseeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicenseeLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if a clause describes a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_affiliate_license-licensee",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 198},
        avg_character_length={"test": 484.11},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADAffiliateLicenseLicensorLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicensorLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause describes a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_affiliate_license-licensor",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 88},
        avg_character_length={"test": 633.40},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADAntiAssignmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAntiAssignmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires consent or notice of a party if the contract is assigned to a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_anti-assignment",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1172},
        avg_character_length={"test": 340.81},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADAuditRightsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAuditRightsLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause gives a party the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_audit_rights",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADCapOnLiabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADCapOnLiabilityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a cap on liability upon the breach of a party's obligation. This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_cap_on_liability",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADChangeOfControlLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADChangeOfControlLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause gives one party the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_change_of_control",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADCompetitiveRestrictionExceptionLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="CUADCompetitiveRestrictionExceptionLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause mentions exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_competitive_restriction_exception",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADCovenantNotToSueLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADCovenantNotToSueLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party is restricted from contesting the validity of the counterparty's ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_covenant_not_to_sue",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADEffectiveDateLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADEffectiveDateLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the agreement becomes effective.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_effective_date",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADExclusivityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADExclusivityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies exclusive dealing commitment with the counterparty. This includes a commitment to procure all 'requirements' from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_exclusivity",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADExpirationDateLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADExpirationDateLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies the date upon which the initial term expires.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_expiration_date",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADGoverningLawLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADGoverningLawLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies which state/country’s law governs the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_governing_law",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2023-08-23", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-4.0",
        socioeconomic_status="high",
        annotations_creators="expert-annotated",
        dialect=[],
        text_creation="found",
        bibtex_citation="""
        @misc{guha2023legalbench,
            title={LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models}, 
            author={Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
            year={2023},
            eprint={2308.11462},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        },
        @article{hendrycks2021cuad,
            title={Cuad: An expert-annotated nlp dataset for legal contract review},
            author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
            journal={arXiv preprint arXiv:2103.06268},
            year={2021}
        }
        """,
        n_samples={"test": 1216},
        avg_character_length={"test": 337.14},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")
