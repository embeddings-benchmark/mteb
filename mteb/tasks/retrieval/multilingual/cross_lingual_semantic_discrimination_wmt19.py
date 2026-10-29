from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_EVAL_LANGS = {
    "deu-fra": ["deu-Latn", "fra-Latn"],
    "fra-deu": ["fra-Latn", "deu-Latn"],
}


class CrossLingualSemanticDiscriminationWMT19(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrossLingualSemanticDiscriminationWMT19",
        dataset={
            "path": "mteb/CrossLingualSemanticDiscriminationWMT19",
            "revision": "5e0eca544ea8d10e317c061ce317178ba94ff195",
        },
        description="Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT19 DE-FR test set",
        reference="https://huggingface.co/datasets/Andrianos/clsd_wmt19_21",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="recall_at_1",
        date=("2018-01-01", "2023-12-12"),
        domains=["News", "Written"],
        task_subtypes=["Cross-Lingual Semantic Discrimination"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="",  # preprint_coming
    )
