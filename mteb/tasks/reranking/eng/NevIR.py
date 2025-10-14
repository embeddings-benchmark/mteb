from mteb._evaluators.retrieval_metrics import paired_accuracy
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class NevIR(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NevIR",
        description="Paired evaluation of real world negation in retrieval, with questions and passages. Since models generally prefer one passage over the other always, there are two questions that the model must get right to understand the negation (hence the `paired_accuracy` metric).",
        reference="https://github.com/orionw/NevIR",
        dataset={
            "path": "orionweller/NevIR-mteb",
            "revision": "eab99575c01c6a8e39f8d2adc6e3c3adcfe84413",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="paired_accuracy",
        date=("2023-05-12", "2023-09-28"),
        domains=["Web"],
        task_subtypes=["Article retrieval"],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Weller2023NevIRNI,
  author = {{Orion Weller and Dawn J Lawrie and Benjamin Van Durme}},
  booktitle = {{Conference of the European Chapter of the Association for Computational Linguistics}},
  title = {{NevIR: Negation in Neural Information Retrieval}},
  url = {{https://api.semanticscholar.org/CorpusID:258676146}},
  year = {{2023}},
}
""",
    )

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return {
            "paired_accuracy": paired_accuracy(qrels, results, scores),
        }
