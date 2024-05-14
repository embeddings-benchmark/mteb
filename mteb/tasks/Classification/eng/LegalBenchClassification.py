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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        date=("1996-01-01", "2023-08-23"),
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
        description="The Corporate Lobbying task consists of determining whether a proposed Congressional bill may be relevant to a company based on a company's self-description in its SEC 10K filing.",
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
        date=("2000-01-01", "2023-08-23"),
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
        date=("2000-01-01", "2023-08-23"),
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
        date=("2000-01-01", "2023-08-23"),
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
        date=("2000-01-01", "2023-08-23"),
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 1246},
        avg_character_length={"test": 375.74},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 416},
        avg_character_length={"test": 391.96},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 220},
        avg_character_length={"test": 433.04},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 308},
        avg_character_length={"test": 402.97},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 236},
        avg_character_length={"test": 277.62},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 762},
        avg_character_length={"test": 369.17},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 876},
        avg_character_length={"test": 309.27},
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
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 876},
        avg_character_length={"test": 289.87},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADInsuranceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADInsuranceLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if clause creates a requirement for insurance that must be maintained by one party for the benefit of the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_insurance",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 1030},
        avg_character_length={"test": 365.54},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADIPOwnershipAssignmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADIPOwnershipAssignmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_ip_ownership_assignment",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 576},
        avg_character_length={"test": 414.00},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADIrrevocableOrPerpetualLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADIrrevocableOrPerpetualLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a license grant that is irrevocable or perpetual.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_irrevocable_or_perpetual_license",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 280},
        avg_character_length={"test": 473.40},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADJointIPOwnershipLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADJointIPOwnershipLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause provides for joint or shared ownership of intellectual property between the parties to the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_joint_ip_ownership",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 192},
        avg_character_length={"test": 374.17},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADLicenseGrantLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADLicenseGrantLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause contains a license granted by one party to its counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_license_grant",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 1396},
        avg_character_length={"test": 409.89},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADLiquidatedDamagesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADLiquidatedDamagesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause awards either party liquidated damages for breach or a fee upon the termination of a contract (termination fee).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_liquidated_damages",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 220},
        avg_character_length={"test": 351.76},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADMinimumCommitmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADMinimumCommitmentLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a minimum order size or minimum amount or units per time period that one party must buy from the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_minimum_commitment",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 772},
        avg_character_length={"test": 364.16},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADMostFavoredNationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADMostFavoredNationLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_most_favored_nation",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 64},
        avg_character_length={"test": 418.75},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNoSolicitOfCustomersLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoSolicitOfCustomersLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_no-solicit_of_customers",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 84},
        avg_character_length={"test": 392.89},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNoSolicitOfEmployeesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoSolicitOfEmployeesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts a party's soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_no-solicit_of_employees",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 142},
        avg_character_length={"test": 417.94},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNonCompeteLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonCompeteLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause restricts the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_non-compete",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 442},
        avg_character_length={"test": 383.20},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNonDisparagementLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonDisparagementLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires a party not to disparage the counterparty.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_non-disparagement",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 100},
        avg_character_length={"test": 403.08},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNonTransferableLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNonTransferableLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause limits the ability of a party to transfer the license being granted to a third party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_non-transferable_license",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 542},
        avg_character_length={"test": 399.16},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADNoticePeriodToTerminateRenewalLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADNoticePeriodToTerminateRenewalLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a notice period required to terminate renewal.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_notice_period_to_terminate_renewal",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 222},
        avg_character_length={"test": 354.85},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADPostTerminationServicesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADPostTerminationServicesLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause subjects a party to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_post-termination_services",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 808},
        avg_character_length={"test": 422.53},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADPriceRestrictionsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADPriceRestrictionsLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause places a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_price_restrictions",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 46},
        avg_character_length={"test": 324.71},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADRenewalTermLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRenewalTermLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a renewal term.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_renewal_term",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 386},
        avg_character_length={"test": 340.87},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADRevenueProfitSharingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRevenueProfitSharingLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause require a party to share revenue or profit with the counterparty for any technology, goods, or services.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_revenue-profit_sharing",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 774},
        avg_character_length={"test": 371.55},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADRofrRofoRofnLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADRofrRofoRofnLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause grant one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_rofr-rofo-rofn",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 690},
        avg_character_length={"test": 395.46},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADSourceCodeEscrowLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADSourceCodeEscrowLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause requires one party to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_source_code_escrow",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 118},
        avg_character_length={"test": 399.18},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADTerminationForConvenienceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADTerminationForConvenienceLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that one party can terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_termination_for_convenience",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 430},
        avg_character_length={"test": 326.30},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADThirdPartyBeneficiaryLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADThirdPartyBeneficiaryLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that that there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_third_party_beneficiary",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 68},
        avg_character_length={"test": 261.04},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADUncappedLiabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADUncappedLiabilityLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies that a party's liability is uncapped upon the breach of its obligation in the contract. This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_uncapped_liability",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 294},
        avg_character_length={"test": 441.04},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADUnlimitedAllYouCanEatLicenseLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause grants one party an “enterprise,” “all you can eat” or unlimited usage license.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_unlimited-all-you-can-eat-license",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 48},
        avg_character_length={"test": 368.08},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADVolumeRestrictionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADVolumeRestrictionLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a fee increase or consent requirement, etc. if one party's use of the product/services exceeds certain threshold.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_volume_restriction",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 322},
        avg_character_length={"test": 306.27},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class CUADWarrantyDurationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADWarrantyDurationLegalBenchClassification",
        description="This task was constructed from the CUAD dataset. It consists of determining if the clause specifies a duration of any warranty against defects or errors in technology, products, or services provided under the contract.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "cuad_warranty_duration",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),
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
        n_samples={"test": 320},
        avg_character_length={"test": 352.27},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class DefinitionClassificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="DefinitionClassificationLegalBenchClassification",
        description="This task consists of determining whether or not a sentence from a Supreme Court opinion offers a definition of a term.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "definition_classification",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
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
        """,
        n_samples={"test": 1337},
        avg_character_length={"test": 253.72},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class Diversity1LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity1LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 1).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_1",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 103.21},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class Diversity2LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity2LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 2).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_2",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 0},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class Diversity3LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity3LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 3).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_3",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 135.46},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class Diversity4LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity4LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 4).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_4",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 144.52},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class Diversity5LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity5LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 5).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_5",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 174.77},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class Diversity6LegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Diversity6LegalBenchClassification",
        description="Given a set of facts about the citizenships of plaintiffs and defendants and the amounts associated with claims, determine if the criteria for diversity jurisdiction have been met (variant 6).",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "diversity_6",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 300},
        avg_character_length={"test": 301.01},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

        # Map the boolean columns to readable plaintext
        _diverse_parties_map = {
            "True": "The parties are diverse.",
            "False": "The parties are not diverse.",
        }

        _amount_in_controversy_map = {
            "True": "The Amount-in-controversy was met.",
            "False": "The Amount-in-controversy was not met.",
        }

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["text"]
                + " "
                + _diverse_parties_map[example["parties_are_diverse"]]
                + " "
                + _amount_in_controversy_map[example["aic_is_met"]]
            }
        )
        self.dataset = self.dataset.remove_columns(
            ["parties_are_diverse", "aic_is_met"]
        )


class FunctionOfDecisionSectionLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FunctionOfDecisionSectionLegalBenchClassification",
        description="""The task is to classify a paragraph extracted from a written court decision into one of seven possible categories:
            1. Facts - The paragraph describes the faction background that led up to the present lawsuit.
            2. Procedural History - The paragraph describes the course of litigation that led to the current proceeding before the court.
            3. Issue - The paragraph describes the legal or factual issue that must be resolved by the court.
            4. Rule - The paragraph describes a rule of law relevant to resolving the issue.
            5. Analysis - The paragraph analyzes the legal issue by applying the relevant legal principles to the facts of the present dispute.
            6. Conclusion - The paragraph presents a conclusion of the court.
            7. Decree - The paragraph constitutes a decree resolving the dispute.
        """,
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "function_of_decision_section",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 367},
        avg_character_length={"test": 551.07},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("answer", "label")

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["Paragraph"]
                + "\n\n"
                + "Citation: "
                + example["Citation"]
            }
        )


class InsurancePolicyInterpretationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="InsurancePolicyInterpretationLegalBenchClassification",
        description="Given an insurance claim and policy, determine whether the claim is covered by the policy.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "insurance_policy_interpretation",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 133},
        avg_character_length={"test": 521.88},
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("answer", "label")

        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["policy"] + "\n\n" + "Claim: " + example["claim"]
            }
        )


class InternationalCitizenshipQuestionsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="InternationalCitizenshipQuestionsLegalBenchClassification",
        description="Answer questions about citizenship law from across the world. Dataset was made using the GLOBALCIT citizenship law dataset, by constructing questions about citizenship law as Yes or No questions.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "international_citizenship_questions",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("1960-01-01", "2023-08-23"),
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
        @misc{vink2023globalcit,
            author = {Vink, Maarten and van der Baaren, Luuk and Bauböck, Rainer and Džankić, Jelena and Honohan, Iseult and Manby, Bronwen},
            title = {GLOBALCIT Citizenship Law Dataset, v2.0, Country-Year-Mode Data (Acquisition)},
            howpublished = {https://hdl.handle.net/1814/73190},
            year = {2023},
            publisher = {Global Citizenship Observatory}
        }
        """,
        n_samples={"test": 2048},
        avg_character_length={"test": 206.18},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_columns(
            {"question": "text", "answer": "label"}
        )
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class JCrewBlockerLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JCrewBlockerLegalBenchClassification",
        description="The J.Crew Blocker, also known as the J.Crew Protection, is a provision included in leveraged loan documents to prevent companies from removing security by transferring intellectual property (IP) into new subsidiaries and raising additional debt. The task consists of detemining whether the J.Crew Blocker is present in the document.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "jcrew_blocker",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2016-01-01", "2023-08-23"),  # best guess
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
        """,
        n_samples={"test": 54},
        avg_character_length={"test": 1092.22},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsBenefitsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsBenefitsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal post discusses public benefits and social services that people can get from the government, like for food, disability, old age, housing, medical help, unemployment, child care, or other social needs.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_benefits",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 66},
        avg_character_length={"test": 1308.44},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsBusinessLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsBusinessLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal question discusses issues faced by people who run small businesses or nonprofits, including around incorporation, licenses, taxes, regulations, and other concerns. It also includes options when there are disasters, bankruptcies, or other problems.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_business",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 174},
        avg_character_length={"test": 1144.51},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsConsumerLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsConsumerLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues people face regarding money, insurance, consumer goods and contracts, taxes, and small claims about quality of service.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_consumer",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 614},
        avg_character_length={"test": 1277.45},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsCourtsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsCourtsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses the logistics of how a person can interact with a lawyer or the court system. It applies to situations about procedure, rules, how to file lawsuits, how to hire lawyers, how to represent oneself, and other practical matters about dealing with these systems.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_courts",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 192},
        avg_character_length={"test": 1171.02},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsCrimeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsCrimeLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues in the criminal system including when people are charged with crimes, go to a criminal trial, go to prison, or are a victim of a crime.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_crime",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 688},
        avg_character_length={"test": 1212.90},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsDivorceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsDivorceLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues around filing for divorce, separation, or annulment, getting spousal support, splitting money and property, and following the court processes.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_divorce",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 150},
        avg_character_length={"test": 1242.43},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsDomesticViolenceLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsDomesticViolenceLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses dealing with domestic violence and abuse, including getting protective orders, enforcing them, understanding abuse, reporting abuse, and getting resources and status if there is abuse.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_domestic_violence",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 174},
        avg_character_length={"test": 1360.83},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsEducationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEducationLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues around school, including accommodations for special needs, discrimination, student debt, discipline, and other issues in education.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_education",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 56},
        avg_character_length={"test": 1397.44},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsEmploymentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEmploymentLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues related to working at a job, including discrimination and harassment, worker's compensation, workers rights, unions, getting paid, pensions, being fired, and more.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_employment",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 710},
        avg_character_length={"test": 1262.74},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsEstatesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsEstatesLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses planning for end-of-life, possible incapacitation, and other special circumstances that would prevent a person from making decisions about their own well-being, finances, and property. This includes issues around wills, powers of attorney, advance directives, trusts, guardianships, conservatorships, and other estate issues that people and families deal with.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_estates",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 178},
        avg_character_length={"test": 1200.70},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsFamilyLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsFamilyLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues that arise within a family, like divorce, adoption, name change, guardianship, domestic violence, child custody, and other issues.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_family",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 2048},
        avg_character_length={"test": 1338.27},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class LearnedHandsHealthLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsHealthLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues with accessing health services, paying for medical care, getting public benefits for health care, protecting one's rights in medical settings, and other issues related to health.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_health",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 226},
        avg_character_length={"test": 1472.59},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsHousingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsHousingLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses issues with paying your rent or mortgage, landlord-tenant issues, housing subsidies and public housing, eviction, and other problems with your apartment, mobile home, or house.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_housing",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 2048},
        avg_character_length={"test": 1322.54},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, splits=["test"]
        )


class LearnedHandsImmigrationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsImmigrationLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's post discusses visas, asylum, green cards, citizenship, migrant work and benefits, and other issues faced by people who are not full citizens in the US.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_immigration",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 134},
        avg_character_length={"test": 1216.31},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsTortsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsTortsLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal question discusses problems that one person has with another person (or animal), like when there is a car accident, a dog bite, bullying or possible harassment, or neighbors treating each other badly.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_torts",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 432},
        avg_character_length={"test": 1406.97},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LearnedHandsTrafficLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LearnedHandsTrafficLegalBenchClassification",
        description="This is a binary classification task in which the model must determine if a user's legal post discusses problems with traffic and parking tickets, fees, driver's licenses, and other issues experienced with the traffic system. It also concerns issues with car accidents and injuries, cars' quality, repairs, purchases, and other contracts.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "learned_hands_traffic",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2022-05-21", "2023-08-23"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @dataset{learned_hands,
            title = {LearnedHands Dataset},
            author = {{Suffolk University Law School} and {Stanford Legal Design Lab}},
            year = {2022},
            url = {https://spot.suffolklitlab.org/data/#learnedhands},
            note = {The LearnedHands dataset is licensed under CC BY-NC-SA 4.0},
            urldate = {2022-05-21}
        }
        """,
        n_samples={"test": 556},
        avg_character_length={"test": 1182.91},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class LegalReasoningCausalityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LegalReasoningCausalityLegalBenchClassification",
        description="Given an excerpt from a district court opinion, classify if it relies on statistical evidence in its reasoning.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "legal_reasoning_causality",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2023-08-23"),  # best guess
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        n_samples={"test": 55},
        avg_character_length={"test": 1563.76},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class SCDBPAccountabilityLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPAccountabilityLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer maintains internal compliance procedures on company standards regarding human trafficking and slavery? This includes any type of internal accountability mechanism. Requiring independently of the supply to comply with laws does not qualify or asking for documentary evidence of compliance does not count either.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "supply_chain_disclosure_best_practice_accountability",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @article{chilton2017limitations,
        title={The limitations of supply chain disclosure regimes},
        author={Chilton, Adam S and Sarfaty, Galit A},
        journal={Stan. J. Int'l L.},
        volume={53},
        pages={1},
        year={2017},
        publisher={HeinOnline}
        }
        """,
        n_samples={"test": 379},
        avg_character_length={"test": 3520},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

class SCDBPAuditsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPAuditsLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "supply_chain_disclosure_best_practice_audits",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @article{chilton2017limitations,
        title={The limitations of supply chain disclosure regimes},
        author={Chilton, Adam S and Sarfaty, Galit A},
        journal={Stan. J. Int'l L.},
        volume={53},
        pages={1},
        year={2017},
        publisher={HeinOnline}
        }
        """,
        n_samples={"test": 379},
        avg_character_length={"test": 3507},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

class SCDBPCertificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPCertificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  performs any type of audit, or reserves the right to audit?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "supply_chain_disclosure_best_practice_certification",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @article{chilton2017limitations,
        title={The limitations of supply chain disclosure regimes},
        author={Chilton, Adam S and Sarfaty, Galit A},
        journal={Stan. J. Int'l L.},
        volume={53},
        pages={1},
        year={2017},
        publisher={HeinOnline}
        }
        """,
        n_samples={"test": 378},
        avg_character_length={"test": 3507},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")

class SCDBPTrainingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPTrainingLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer  provides training to employees on human trafficking and slavery? Broad policies such as ongoing dialogue on mitigating risks of human trafficking and slavery or increasing managers and purchasers knowledge about health, safety and labor practices qualify as training. Providing training to contractors who failed to comply with human trafficking laws counts as training.'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "supply_chain_disclosure_best_practice_training",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @article{chilton2017limitations,
        title={The limitations of supply chain disclosure regimes},
        author={Chilton, Adam S and Sarfaty, Galit A},
        journal={Stan. J. Int'l L.},
        volume={53},
        pages={1},
        year={2017},
        publisher={HeinOnline}
        }
        """,
        n_samples={"test": 379},
        avg_character_length={"test": 3506},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")


class SCDBPVerificationLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SCDBPVerificationLegalBenchClassification",
        description="This is a binary classification task in which the LLM must determine if a supply chain disclosure meets the following coding criteria: 'Does the above statement disclose whether the retail seller or manufacturer engages in verification and auditing as one practice, expresses that it may conduct an audit, or expressess that it is assessing supplier risks through a review of the US Dept. of Labor's List?'",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "nguha/legalbench",
            "name": "supply_chain_disclosure_best_practice_verification",
            "revision": "12ca3b695563788fead87a982ad1a068284413f4",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2010-01-01", "2015-06-30"),
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
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
        @article{chilton2017limitations,
        title={The limitations of supply chain disclosure regimes},
        author={Chilton, Adam S and Sarfaty, Galit A},
        journal={Stan. J. Int'l L.},
        volume={53},
        pages={1},
        year={2017},
        publisher={HeinOnline}
        }
        """,
        n_samples={"test": 379},
        avg_character_length={"test": 3498},
    )

    def dataset_transform(self):
        mapping = {"yes": 1, "no": 0}
        self.dataset = self.dataset.map(
            lambda example: {
                "answer": mapping.get(example["answer"].lower(), example["answer"])
            }
        )
        self.dataset = self.dataset.rename_column("answer", "label")
