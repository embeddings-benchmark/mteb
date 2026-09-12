from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/waxal-audio-text-retrieval"
_DATASET_REVISION = "f57c8af71955625a1dd49b3cf552a23e3e65dafb"

# Keys are WAXAL's own directory names; values are ISO 639-3 plus the script the
# transcriptions use. Two of WAXAL's short names collide with unrelated ISO codes and
# are deliberately not passed through - see the comments on `mas` and `sog`.
_WAXAL_LANGS = {
    "ach": ["ach-Latn"],  # Acholi
    "aka": ["aka-Latn"],  # Akan
    "amh": ["amh-Ethi"],  # Amharic
    "ewe": ["ewe-Latn"],  # Ewe
    "ful": ["ful-Latn"],  # Fula
    "lin": ["lin-Latn"],  # Lingala
    "lug": ["lug-Latn"],  # Luganda
    # WAXAL's "mas" is Masaaba (Lumasaaba, Uganda), not Maasai; ISO 639-3 `mas` is Maasai
    "mas": ["myx-Latn"],
    "mlg": ["mlg-Latn"],  # Malagasy
    "nyn": ["nyn-Latn"],  # Nyankole
    "orm": ["orm-Latn"],  # Oromo, written in Latin (Qubee)
    "sid": ["sid-Latn"],  # Sidama, written in Latin since 1993
    "sna": ["sna-Latn"],  # Shona
    # WAXAL's "sog" is Soga (Lusoga, Uganda); ISO 639-3 `sog` is Sogdian, an extinct
    # Iranian language, so the Lusoga code `xog` is used instead
    "sog": ["xog-Latn"],
    "tir": ["tir-Ethi"],  # Tigrinya
    "wal": ["wal-Ethi"],  # Wolaytta
}

_BIBTEX = r"""
@misc{diack2026waxal,
  archiveprefix = {arXiv},
  author = {Diack, Abdoulaye and Nelson, Perry and Agbesi, Kwaku and Nakalembe, Angela and MohamedKhair, MohamedElfatih and Dube, Vusumuzi and Siyavora, Tavonga and Venugopalan, Subhashini and Hickey, Jason and Okonkwo, Uche and Bapna, Abhishek and Wiafe, Isaac and Helegah, Raynard Dodzi and Atsakpo, Elikem Doe and Nutrokpor, Charles and Winful, Fiifi Baffoe Payin and Solaga, Kafui Kwashie and Abdulai, Jamal-Deen and Ekpezu, Akon Obu and Niyonkuru, Audace and Rutunda, Samuel and Ishimwe, Boris and Melese, Michael and Bainomugisha, Engineer and Nakatumba-Nabende, Joyce and Katumba, Andrew and Babirye, Claire and Mukiibi, Jonathan and Kimani, Vincent and Kibacia, Samuel and Maina, James and Emmah, Fridah and Shekarau, Ahmed Ibrahim and Adamu, Ibrahim Shehu and Abdullahi, Yusuf and Lakougna, Howard and MacDonald, Bob and Shemtov, Hadar and Walcott-Bryant, Aisha and Cisse, Moustapha and Hassidim, Avinatan and Dean, Jeff and Matias, Yossi},
  eprint = {2602.02734},
  title = {{WAXAL}: A Large-Scale Multilingual African Language Speech Corpus},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Multilingual speech-text retrieval over WAXAL, a corpus of image-prompted natural "
    "speech in Sub-Saharan African languages. Most of these languages have no presence in "
    "mteb's existing multilingual audio tasks, which cover mainly European and South/East "
    "Asian languages. Each subset is scored against its own transcript pool."
)


def _load_waxal(task: AbsTaskRetrieval, direction: str) -> None:
    """Shared loader for both directions."""
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    task.dataset = {}

    for lang in task.hf_subsets:
        ds = load_dataset(_DATASET_PATH, lang, revision=_DATASET_REVISION, split=split)
        audio_ds = ds.select_columns(["id", "audio"])
        text_ds = ds.select_columns(["id", "text"])
        ids = ds["id"]

        if direction == "a2t":
            queries, corpus = audio_ds, text_ds
        else:
            queries, corpus = text_ds, audio_ds

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs={i: {i: 1} for i in ids},
                top_ranked=None,
            )
        }

    task.data_loaded = True


class WaxalA2TRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WaxalA2TRetrieval",
        description=f"{_DESCRIPTION} Retrieve the transcription of a spoken utterance.",
        reference="https://arxiv.org/abs/2602.02734",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2t",
        eval_splits=["test"],
        eval_langs=_WAXAL_LANGS,
        main_score="hit_rate_at_5",
        modalities=["audio", "text"],
        date=("2024-01-01", "2026-02-28"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the transcription of this spoken utterance."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_waxal(self, "a2t")


class WaxalT2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WaxalT2ARetrieval",
        description=f"{_DESCRIPTION} Retrieve the recording matching a transcription.",
        reference="https://arxiv.org/abs/2602.02734",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="t2a",
        eval_splits=["test"],
        eval_langs=_WAXAL_LANGS,
        main_score="hit_rate_at_5",
        modalities=["text", "audio"],
        date=("2024-01-01", "2026-02-28"),
        domains=["Spoken"],
        task_subtypes=["Speech Transcription Retrieval"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the recording of the following transcription."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_waxal(self, "t2a")
