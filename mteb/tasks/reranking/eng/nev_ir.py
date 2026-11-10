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
@inproceedings{weller_nevir_2024,
  title = {{N}ev{IR}: Negation in Neural Information Retrieval},
  author = {Weller, Orion  and
      Lawrie, Dawn  and
      Van Durme, Benjamin},
  editor = {Graham, Yvette  and
      Purver, Matthew},
  booktitle = {Proceedings of the 18th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)},
  month = mar,
  year = {2024},
  address = {St. Julian{'}s, Malta},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2024.eacl-long.139/},
  doi = {10.18653/v1/2024.eacl-long.139},
  pages = {2274--2287},
  abstract = {Negation is a common everyday phenomena and has been a consistent area of weakness for language models (LMs). Although the Information Retrieval (IR) community has adopted LMs as the backbone of modern IR architectures, there has been little to no research in understanding how negation impacts neural IR. We therefore construct a straightforward benchmark on this theme: asking IR models to rank two documents that differ only by negation. We show that the results vary widely according to the type of IR architecture: cross-encoders perform best, followed by late-interaction models, and in last place are bi-encoder and sparse neural architectures. We find that most current information retrieval models do not consider negation, performing similarly or worse than randomly ranking. We show that although the obvious approach of continued fine-tuning on a dataset of contrastive documents containing negations increases performance (as does model size), there is still a large gap between machine and human performance.},
}
""",
    )

    def task_specific_scores(  # noqa: PLR6301
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
