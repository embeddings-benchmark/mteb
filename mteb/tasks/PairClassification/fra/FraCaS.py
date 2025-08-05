from datasets import load_dataset
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

class FracasTask(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="fracas",
        description=(
            "Natural language inference on FRACAS: "
            "prédit si une hypothèse découle (entailment) ou non d'une prémisse."
        ),
        reference="https://huggingface.co/datasets/maximoss/fracas",
        dataset={"path": "maximoss/fracas", "revision": "main"},
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["train"],        # FRACAS ne propose que ce split
        eval_langs=["fra-Latn"],
        main_score="max_accuracy",
        date=("2025-08-05", "2025-08-05"),
        domains=["Academic"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fracas2025,
  author    = {Maxim Oss and Collaborateurs},
  title     = {FRACAS: A French NLI dataset},
  booktitle = {Imaginary Conference on French NLP},
  year      = {2025},
}
""",
    )

    def load_data(self, **kwargs):
        """Charge le DatasetDict HF puis transforme en self.dataset."""
        if getattr(self, "data_loaded", False):
            return
        self.dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
        )
        self.dataset_transform()
        self.data_loaded = True

    def dataset_transform(self):
        """
        Construit self.dataset sous la forme :
        {
          'fra-Latn': {
             'train': [
               {
                 'sentence1': [...],  # liste de prémisses
                 'sentence2': [...],  # liste d’hypothèses
                 'labels':    [...],  # liste de 0/1
               }
             ]
          }
        }
        """
        out: dict[str, dict[str, list[dict[str, list]]]] = {}
        for lang in self.hf_subsets:                 # ['fra-Latn']
            out[lang] = {}
            for split in self.metadata.eval_splits:  # ['train']
                ds = self.dataset[split]
                # Affiche les labels pour debugging
                print("FRACAS labels disponibles :", sorted(set(ds["label"])))
                # Filtrer hors 'undef'
                ds = ds.filter(lambda x: x["label"] != "undef")
                # Remapper '1'→1 (positif), '0' et '2'→0 (négatif)
                ds = ds.map(lambda ex: {"label": 1 if ex["label"] == "1" else 0})
                # Construire la liste contenant UN dict de listes alignées
                out[lang][split] = [
                    {
                        "sentence1": ds["premises"],
                        "sentence2": ds["hypothesis"],
                        "labels":    ds["label"],
                    }
                ]
        self.dataset = out