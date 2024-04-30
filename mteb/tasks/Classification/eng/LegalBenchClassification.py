from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class CanadaTaxCourtOutcomesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CanadaTaxCourtOutcomesLegalBenchClassification",
        description="Canada Tax Court Outcomes LegalBench Classification Dataset",
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
        description="Contract NLI Confidentiality Of Agreement LegalBench Classification Dataset",
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
        description="Contract NLI Explicit Identification LegalBench Classification Dataset",
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


class CUADAuditRightsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAuditRightsLegalBenchClassification",
        description="CUAD Audit Rights LegalBench Classification Dataset",
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


class CUADAntiAssignmentLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAntiAssignmentLegalBenchClassification",
        description="CUAD Anti-Assignment LegalBench Classification Dataset",
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


class CUADAffiliateLicenseLicensorLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicensorLegalBenchClassification",
        description="CUAD Affiliate License-Licensor LegalBench Classification Dataset",
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


class CUADAffiliateLicenseLicenseeLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CUADAffiliateLicenseLicenseeLegalBenchClassification",
        description="CUAD Affiliate License-Licensee LegalBench Classification Dataset",
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


class CorporateLobbyingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="CorporateLobbyingLegalBenchClassification",
        description="Corporate Lobbying LegalBench Classification Dataset",
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


class ContractNLISurvivalOfObligationsLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISurvivalOfObligationsLegalBenchClassification",
        description="Contract NLI Survival of Obligations LegalBench Classification Dataset",
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


class ContractNLISharingWithThirdPartiesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithThirdPartiesLegalBenchClassification",
        description="Contract NLI Sharing With Third Parties LegalBench Classification Dataset",
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


class ContractNLISharingWithEmployeesLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLISharingWithEmployeesLegalBenchClassification",
        description="Contract NLI Sharing With Employees LegalBench Classification Dataset",
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


class ContractNLIReturnOfConfidentialInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIReturnOfConfidentialInformationLegalBenchClassification",
        description="Contract NLI Return of Confidential Information LegalBench Classification Dataset",
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


class ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissiblePostAgreementPossessionLegalBenchClassification",
        description="Contract NLI Permissible Post Agreement Possession LegalBench Classification Dataset",
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


class ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleDevelopmentOfSimilarInformationLegalBenchClassification",
        description="Contract NLI Permissible Development of Similar Information LegalBench Classification Dataset",
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


class ContractNLIPermissibleCopyLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleCopyLegalBenchClassification",
        description="Contract NLI Permissible Copy LegalBench Classification Dataset",
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


class ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIPermissibleAcquirementOfSimilarInformationLegalBenchClassification",
        description="Contract NLI Permissible Acquirement of Similar Information LegalBench Classification Dataset",
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


class ContractNLINoticeOnCompelledDisclosureLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLINoticeOnCompelledDisclosureLegalBenchClassification",
        description="Contract NLI Notice on Compelled Disclosure LegalBench Classification Dataset",
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


class ContractNLINoLicensingLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLINoLicensingLegalBenchClassification",
        description="Contract NLI No Licensing LegalBench Classification Dataset",
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


class ContractNLILimitedUseLegalBenchClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ContractNLILimitedUseLegalBenchClassification",
        description="Contract NLI Limited Use LegalBench Classification Dataset",
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


class ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification(
    AbsTaskClassification
):
    metadata = TaskMetadata(
        name="ContractNLIInclusionOfVerballyConveyedInformationLegalBenchClassification",
        description="Contract NLI Inclusion of Verbally Conveyed Information LegalBench Classification Dataset",
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
