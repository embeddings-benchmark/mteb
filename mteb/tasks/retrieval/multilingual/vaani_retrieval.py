from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

_DATASET_PATH = "vnahata/vaani-audio-image-retrieval"
_DATASET_REVISION = "949ed35276facabd185332c118a473ca8ec5dd79"

# Vaani groups its recordings by the language the speaker used. Each key is the
# collection's own language label; the value is the ISO 639-3 code plus the script
# its transcripts use. Codes marked "approx." have no distinct ISO 639-3 entry and
# are mapped to the closest coded variety.
_VAANI_LANGS = {
    "Angami": ["njm-Latn"],
    "Angika": ["anp-Deva"],
    "Ao": ["njo-Latn"],
    "Awadhi": ["awa-Deva"],
    "Bagheli": ["bfy-Deva"],
    "Bagri": ["bgq-Deva"],
    "Bajjika": ["mai-Deva"],  # approx: no distinct ISO 639-3 entry
    "Bearybashe": ["kan-Knda"],  # approx: no distinct ISO 639-3 entry
    "Bhatri": ["bgw-Deva"],
    "Bhili": ["bhb-Deva"],
    "Bundeli": ["bns-Deva"],
    "Chakhesang": ["nbe-Latn"],
    "Chhattisgarhi": ["hne-Deva"],
    "Desia": ["dso-Orya"],
    "Garhwali": ["gbm-Deva"],
    "Gondi": ["gon-Deva"],
    "Halbi": ["hlb-Deva"],
    "Haryanvi": ["bgc-Deva"],
    "Idu Mishmi": ["clk-Latn"],
    "Jaipuri": ["dhd-Deva"],  # approx: no distinct ISO 639-3 entry
    "Karbi": ["mjw-Latn"],
    "Kashmiri": ["kas-Arab"],
    "Khandeshi": ["khn-Deva"],
    "Khariboli": ["hin-Deva"],  # approx: no distinct ISO 639-3 entry
    "Khortha": ["mag-Deva"],  # approx: no distinct ISO 639-3 entry
    "Kokborok": ["trp-Latn"],
    "Konkani": ["kok-Deva"],
    "Koya": ["kff-Telu"],
    "Kumaoni": ["kfy-Deva"],
    "Kurmali": ["kyw-Deva"],
    "Kurukh": ["kru-Deva"],
    "Lambani": ["lmn-Deva"],
    "Lepcha": ["lep-Lepc"],
    "Lotha": ["njh-Latn"],
    "Magahi": ["mag-Deva"],
    "Malayalam": ["mal-Mlym"],
    "Malvani": ["kok-Deva"],  # approx: no distinct ISO 639-3 entry
    "Malvi": ["mup-Deva"],
    "Nimadi": ["noe-Deva"],
    "Nyishi": ["njz-Latn"],
    "Pahadi": ["kfx-Deva"],  # approx: no distinct ISO 639-3 entry
    "Powari": ["pwr-Deva"],
    "Rengma": ["nre-Latn"],
    "Rongmei": ["nbu-Latn"],
    "Sadri": ["sck-Deva"],
    "Sambalpuri": ["spv-Orya"],
    "Sangtam": ["nsa-Latn"],
    "Santali": ["sat-Olck"],
    "Shekhawati": ["swv-Deva"],
    "Sikkimese": ["sip-Tibt"],
    "Sindhi": ["snd-Arab"],
    "Sumi": ["nsm-Latn"],
    "Surgujia": ["sgj-Deva"],
    "Surjapuri": ["sjp-Deva"],
    "Sylheti": ["syl-Sylo"],
    "Tagin": ["tgj-Latn"],
    "Tangkhul": ["nmf-Latn"],
    "Tulu": ["tcy-Knda"],
    "Urdu": ["urd-Arab"],
    "Wancho": ["nnp-Latn"],
    "Yimchunger": ["yim-Latn"],
    "Zeme": ["nzm-Latn"],
}

_BIBTEX = r"""
@misc{pulikodan2026vaani,
  archiveprefix = {arXiv},
  author = {Pulikodan, Sujith and Singh, Abhayjeet and Basu, Agneedh and Desai, Nihar and J, Pavan Kumar and Bhat, Pranav D and Dharmaraju, Raghu and Gupta, Ritika and Udupa, Sathvik and Kumar, Saurabh and Sharma, Sumit and Sanka, Visruth and Tewari, Dinesh and Dhand, Harsh and Kamat, Amrita and Singh, Sukhwinder and Vashishth, Shikhar and Talukdar, Partha and Acharya, Raj and Ghosh, Prasanta Kumar},
  eprint = {2603.28714},
  title = {VAANI: Capturing the language landscape for an inclusive digital India},
  year = {2026},
}
"""

_DESCRIPTION = (
    "Multilingual audio-image retrieval built from Project Vaani, an India-wide "
    "collection of spontaneous image-prompted speech. Speakers were shown a photograph "
    "and asked to describe it in their own language, so each recording is grounded in a "
    "specific image."
)


def _load_vaani(task: AbsTaskRetrieval, direction: str) -> None:
    """Shared loader for both Vaani retrieval directions.

    a2i scores every language subset against one shared image corpus, so subset
    differences reflect the query language rather than corpus size. Each recording
    describes exactly one image, so the labels stay unambiguous.

    i2a keeps the corpus per language. A shared recording pool would be wrong here:
    709 of the 3,552 images are described in more than one language, so an image query
    would have correct answers in other subsets sitting in the corpus labelled
    irrelevant.
    """
    if task.data_loaded:
        return

    split = task.metadata.eval_splits[0]
    images = load_dataset(
        _DATASET_PATH, "images", revision=_DATASET_REVISION, split=split
    ).select_columns(["id", "image"])

    per_lang = {
        lang: load_dataset(_DATASET_PATH, lang, revision=_DATASET_REVISION, split=split)
        for lang in task.hf_subsets
    }

    task.dataset = {}
    for lang, audio in per_lang.items():
        # Read the link columns without touching the media columns: iterating full rows
        # would decode every waveform or image just to collect two strings.
        links = audio.select_columns(["id", "image_id"]).to_dict()
        pairs = list(zip(links["id"], links["image_id"], strict=True))

        if direction == "a2i":
            queries = audio.select_columns(["id", "audio"])
            corpus = images
            qrels = {qid: {img: 1} for qid, img in pairs}
        else:
            qrels = {}
            for qid, img in pairs:
                qrels.setdefault(img, {})[qid] = 1
            keep = set(qrels)
            # select() by index rather than filter(), which would decode every image
            wanted = [i for i, id_ in enumerate(images["id"]) if id_ in keep]
            queries = images.select(wanted)
            corpus = audio.select_columns(["id", "audio"])

        task.dataset[lang] = {
            split: RetrievalSplitData(
                queries=queries,
                corpus=corpus,
                relevant_docs=qrels,
                top_ranked=None,
            )
        }

    task.data_loaded = True


class VaaniA2IRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VaaniA2IRetrieval",
        description=(
            f"{_DESCRIPTION} Given a spoken description, retrieve the photograph it "
            "describes. Every language subset is scored against the same image corpus, "
            "so differences between subsets reflect the query language rather than "
            "corpus difficulty."
        ),
        reference="https://arxiv.org/abs/2603.28714",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="a2i",
        eval_splits=["test"],
        eval_langs=_VAANI_LANGS,
        main_score="ndcg_at_10",
        modalities=["audio", "image"],
        date=("2023-01-01", "2025-06-30"),
        domains=["Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the image that this spoken description refers to."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vaani(self, "a2i")


class VaaniI2ARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="VaaniI2ARetrieval",
        description=(
            f"{_DESCRIPTION} Given a photograph, retrieve the spoken descriptions of it. "
            "The corpus is per language: 709 of the images are described in more than "
            "one language, so a pooled corpus would leave correct answers from other "
            "subsets labelled irrelevant."
        ),
        reference="https://arxiv.org/abs/2603.28714",
        dataset={"path": _DATASET_PATH, "revision": _DATASET_REVISION},
        type="Any2AnyRetrieval",
        category="i2a",
        eval_splits=["test"],
        eval_langs=_VAANI_LANGS,
        main_score="ndcg_at_10",
        modalities=["audio", "image"],
        date=("2023-01-01", "2025-06-30"),
        domains=["Spoken"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_BIBTEX,
        prompt={"query": "Find the spoken descriptions of this image."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        _load_vaani(self, "i2a")
