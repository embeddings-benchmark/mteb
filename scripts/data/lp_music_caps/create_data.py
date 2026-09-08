"""Create the LPMusicCapsMTT retrieval datasets.

Source: mulab-mir/lp-music-caps-magnatagatune-3k, `test` split (300 clips).
Each clip is 10s of MagnaTagATune audio at 22.05 kHz with four LLM-generated
captions from LP-MusicCaps (Doh et al., ISMIR 2023).

Produces two repos, each with `corpus`, `queries` and `default` (qrels) configs
on a `test` split:
    LPMusicCapsMTT_a2t   audio query   -> caption corpus (4 relevant per query)
    LPMusicCapsMTT_t2a   caption query -> audio corpus   (1 relevant per query)
"""

import os

from datasets import Audio, Dataset, DatasetDict, load_dataset

WRITE_TOK = os.environ["HF_TOKEN"]
NAMESPACE = "hubxrt"
SOURCE = "mulab-mir/lp-music-caps-magnatagatune-3k"
SPLIT = "test"

ds = load_dataset(SOURCE, split=SPLIT)
track_ids = [str(t) for t in ds["track_id"]]
if len(set(track_ids)) != len(track_ids):
    raise ValueError("duplicate track_ids would collide as document ids")

# Audio side. Cast to Audio() so it decodes to {"array", "sampling_rate"},
# which is the shape mteb's collator and statistics code expect.
audio_ds = (
    ds.select_columns(["audio"])
    .add_column("_id", track_ids)
    .cast_column("audio", Audio())
)

# Caption side. Four captions per clip, id'd as "{track_id}_{i}".
caption_ids: list[str] = []
caption_texts: list[str] = []
pairs: list[tuple[str, str]] = []
for track_id, texts in zip(track_ids, ds["texts"]):
    for i, text in enumerate(texts):
        caption_ids.append(f"{track_id}_{i}")
        caption_texts.append(text)
        pairs.append((track_id, f"{track_id}_{i}"))

caption_ds = Dataset.from_dict({"_id": caption_ids, "text": caption_texts})


def qrels(query_ids: list[str], corpus_ids: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "query-id": query_ids,
            "corpus-id": corpus_ids,
            "score": [1] * len(query_ids),
        }
    )


datasets = {
    f"{NAMESPACE}/LPMusicCapsMTT_a2t": {
        "queries": audio_ds,
        "corpus": caption_ds,
        "default": qrels([t for t, _ in pairs], [c for _, c in pairs]),
    },
    f"{NAMESPACE}/LPMusicCapsMTT_t2a": {
        "queries": caption_ds,
        "corpus": audio_ds,
        "default": qrels([c for _, c in pairs], [t for t, _ in pairs]),
    },
}

for repo, configs in datasets.items():
    for config, dataset in configs.items():
        DatasetDict({SPLIT: dataset}).push_to_hub(
            repo, config_name=config, token=WRITE_TOK
        )
    print(f"pushed {repo}")
