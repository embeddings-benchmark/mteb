from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES = {
    "wmt21.de.fr": ["deu-Latn", "fra-Latn"],
    "wmt21.fr.de": ["fra-Latn", "deu-Latn"],
}


def _build_lang_pair(langs: list[str]) -> str:
    """Builds a language pair separated by a dash.
    e.g., ['eng-Latn', 'deu-Latn'] -> 'eng-deu'.
    """
    return langs[0].split("-")[0] + "-" + langs[1].split("-")[0]


def extend_lang_pairs() -> dict[str, list[str]]:
    eval_langs = {}
    for langs in _LANGUAGES.values():
        lang_pair = _build_lang_pair(langs)
        eval_langs[lang_pair] = langs
    return eval_langs


_EVAL_LANGS = extend_lang_pairs()


class CrossLingualSemanticDiscriminationWMT21(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrossLingualSemanticDiscriminationWMT21",
        dataset={
            "path": "mteb/CrossLingualSemanticDiscriminationWMT21",
            "revision": "0d23cec1d8fcab64c43df8e4182b4bcfe0cf72d7",
        },
        description="Evaluate a multilingual embedding model based on its ability to discriminate against the original parallel pair against challenging distractors - spawning from WMT21 DE-FR test set",
        reference="https://huggingface.co/datasets/Andrianos/clsd_wmt19_21",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="recall_at_1",
        date=("2020-01-01", "2023-12-12"),
        domains=["News", "Written"],
        task_subtypes=["Cross-Lingual Semantic Discrimination"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="",  # preprint_coming
    )

    num_of_distractors = 4
