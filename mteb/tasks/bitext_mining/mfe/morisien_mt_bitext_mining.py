from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_EVAL_LANGS = {
    "mfe_Latn-eng_Latn": ["mfe-Latn", "eng-Latn"],
    "eng_Latn-mfe_Latn": ["eng-Latn", "mfe-Latn"],
    "mfe_Latn-fra_Latn": ["mfe-Latn", "fra-Latn"],
    "fra_Latn-mfe_Latn": ["fra-Latn", "mfe-Latn"],
}


class MorisienMTBitextMining(AbsTaskBitextMining):
    metadata = TaskMetadata(
        name="MorisienMTBitextMining",
        dataset={
            "path": "mteb/MorisienMTBitextMining",
            "revision": "45f511e86cc511a422a130f2d23a2697e279efa2",
        },
        description=(
            "Machine translation test set aligning Mauritian Creole (Kreol Morisien) with English and French, "
            "from the held-out test split of MorisienMT."
        ),
        reference="https://arxiv.org/abs/2206.02421",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_EVAL_LANGS,
        main_score="f1",
        date=(
            "2010-01-01",
            "2022-06-01",
        ),  # estimated: aggregated from earlier published translations
        domains=["Written", "Religious", "Government", "Fiction"],
        task_subtypes=[],
        license="mit",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{dabre2022morisienmt,
  author = {Dabre, Raj and Sukhoo, Aneerav},
  journal = {arXiv preprint arXiv:2206.02421},
  title = {MorisienMT: A Dataset for Mauritian Creole Machine Translation},
  year = {2022},
}
""",
    )
