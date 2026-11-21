from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class HumanEvalRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HumanEvalRetrieval",
        description="A code retrieval task based on 164 Python programming problems from HumanEval. Each query is a natural language description of a programming task (e.g., 'Check if in given list of numbers, are any two numbers closer to each other than given threshold'), and the corpus contains Python code implementations. The task is to retrieve the correct code snippet that solves the described problem. Queries are problem descriptions while the corpus contains Python function implementations with proper indentation and logic.",
        reference="https://huggingface.co/datasets/embedding-benchmark/HumanEval",
        dataset={
            "path": "mteb/HumanEvalRetrieval",
            "revision": "db44dbc45b006cfb9588cde47e4408ad8022e836",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{chen2021evaluating,
  archiveprefix = {arXiv},
  author = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and Ray, Alex and Puri, Raul and Krueger, Gretchen and Petrov, Michael and Khlaaf, Heidy and Sastry, Girish and Mishkin, Pamela and Chan, Brooke and Gray, Scott and Ryder, Nick and Pavlov, Mikhail and Power, Alethea and Kaiser, Lukasz and Bavarian, Mohammad and Winter, Clemens and Tillet, Philippe and Such, Felipe Petroski and Cummings, Dave and Plappert, Matthias and Chantzis, Fotios and Barnes, Elizabeth and Herbert-Voss, Ariel and Guss, William Hebgen and Nichol, Alex and Paino, Alex and Tezak, Nikolas and Tang, Jie and Babuschkin, Igor and Balaji, Suchir and Jain, Shantanu and Saunders, William and Hesse, Christopher and Carr, Andrew N. and Leike, Jan and Achiam, Joshua and Misra, Vedant and Morikawa, Evan and Radford, Alec and Knight, Matthew and Brundage, Miles and Murati, Mira and Mayer, Katie and Welinder, Peter and McGrew, Bob and Amodei, Dario and McCandlish, Sam and Sutskever, Ilya and Zaremba, Wojciech},
  eprint = {2107.03374},
  primaryclass = {cs.LG},
  title = {Evaluating Large Language Models Trained on Code},
  year = {2021},
}""",
    )
