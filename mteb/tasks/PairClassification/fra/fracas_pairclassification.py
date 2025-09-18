from __future__ import annotations

from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FracasPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="FracasPairClassification",
        description=(
            "Natural language inference on FRACAS: "
            "prédit si une hypothèse découle (entailment) ou non d'une prémisse."
        ),
        reference="https://huggingface.co/datasets/maximoss/fracas",
        dataset={
            "path": "maximoss/fracas",
            "revision": "2506e60be409b124bd72336038dea6f9460ea70c",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["fra-Latn"],
        main_score="max_accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Academic"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{amblard-etal-2020-french,
    title = "A {F}rench Version of the {F}ra{C}a{S} Test Suite",
    author = "Amblard, Maxime  and
      Beysson, Cl{\'e}ment  and
      de Groote, Philippe  and
      Guillaume, Bruno  and
      Pogodalla, Sylvain",
    editor = "Calzolari, Nicoletta  and
      B{\'e}chet, Fr{\'e}d{\'e}ric  and
      Blache, Philippe  and
      Choukri, Khalid  and
      Cieri, Christopher  and
      Declerck, Thierry  and
      Goggi, Sara  and
      Isahara, Hitoshi  and
      Maegaard, Bente  and
      Mariani, Joseph  and
      Mazo, H{\'e}l{\`e}ne  and
      Moreno, Asuncion  and
      Odijk, Jan  and
      Piperidis, Stelios",
    booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.721",
    pages = "5887--5895",
    language = "English",
    ISBN = "979-10-95546-34-4",
}

""",
    )

    def dataset_transform(self):
        out: dict[str, dict[str, list[dict[str, list]]]] = {}
        for lang in self.hf_subsets:
            out[lang] = {}
            for split in self.metadata.eval_splits:
                ds = self.dataset[split]
                ds = ds.filter(lambda x: x["label"] != "undef")
                ds = ds.map(lambda ex: {"label": 1 if ex["label"] == "1" else 0})
                out[lang][split] = [
                    {
                        "sentence1": ds["premises"],
                        "sentence2": ds["hypothesis"],
                        "labels": ds["label"],
                    }
                ]
        self.dataset = out
