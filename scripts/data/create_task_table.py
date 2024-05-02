from __future__ import annotations

import os

from mteb import MTEB

HEADER = "| Name | Hub URL | Description | Type | Category | #Languages | Train #Samples | Dev #Samples | Test #Samples | Avg. chars / train | Avg. chars / dev | Avg. chars / test"
SEP = "|:-----|:-----|:-----|:-----|:-----|-----:|-----:|-----:|-----:|-----:|-----:|-----:|"
ONE_LINE = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |"

TABLE_STRING = "\n".join([HEADER, SEP])

LEN_KEYS = {
    "text",
    "sentences",
    "sentence1",
    "sentence2",
    "sent1",
    "sent2" "query",
    "positive",
    "negative" "queries",
    "corpus",
    "machine_summaries",
    "human_summaries",
}


DATAPATH = "/gpfsscratch/rech/six/commun/commun/experiments/muennighoff/mteb"


def load_data(hf_hub_name, subset=None):
    """Load dataset from Hub via cloning for easy offline usage with HF_DATASETS_OFFLINE=1
    Can be replaced with just `load_dataset(hf_hub_name, subset)` if preferred
    """
    from datasets import load_dataset

    path = os.path.join(DATAPATH, hf_hub_name)
    if os.path.exists(path):
        dataset = load_dataset(path, subset)
    else:
        from git import Repo

        Repo.clone_from("https://huggingface.co/datasets/" + hf_hub_name, path)
        dataset = load_dataset(path, subset)
    return dataset


def get_ds_stats_beir_hub(hf_hub_name):
    """Not used as some BEIR datasets are still missing on the Hub"""
    lens = {}
    for subset in ["corpus", "queries"]:
        ds = load_data("mteb/hfbeir" + hf_hub_name.replace("BeIR", ""), subset)
        splits = list(ds.keys())
        len_keys = set(ds[splits[-1]].features.keys()) & LEN_KEYS
        for split in splits:
            if split not in ds:
                continue
            lens.setdefault(split, [])
            for k in len_keys:
                if isinstance(ds[split][k][0], str):
                    lens[split] += [len(x) for x in ds[split][k]]
                elif isinstance(ds[split][k][0], list):
                    assert isinstance(ds[split][k][0][0], str), f"Too nested: {k}"
                    lens[split] += [len(y) for x in ds[split][k] for y in x]
                else:
                    raise ValueError(f"Unknown type {type(ds[split][k])}")
    all_lens = [x for y in lens.values() for x in y]
    avg_len = sum(all_lens) / len(all_lens)
    return ["TODO"] * 3 + [round(avg_len, 1)] * 3


def get_ds_stats_beir(hf_hub_name):
    from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader

    path = os.path.join(DATAPATH, hf_hub_name)
    if not os.path.exists(path):
        from beir import util

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{hf_hub_name}.zip"
        util.download_and_unzip(url, DATAPATH)
    lens = {"train": [], "dev": [], "test": []}
    for split in lens.keys():
        try:
            corpus, queries, relevant_docs = BeirDataLoader(path).load(split=split)
        except:  # split does not exist  # noqa: E722
            continue
        # + 1 for space added between Title & Text by default in BEIR
        avg_lens_c = [len(v["text"]) + len(v["title"]) + 1 for v in corpus.values()]
        avg_lens_q = [len(v) for v in queries.values()]
        lens[split].extend(avg_lens_c)
        lens[split].extend(avg_lens_q)
    avg_lens = {
        k: round(sum(lens[k]) / len(lens[k]), 1) if lens[k] else 0 for k in lens
    }
    return (
        len(lens["train"]),
        len(lens["dev"]),
        len(lens["test"]),
        avg_lens["train"],
        avg_lens["dev"],
        avg_lens["test"],
    )


def get_ds_stats(hf_hub_name):
    ds = load_data(hf_hub_name)
    assert "test" in ds, f"No test set for {hf_hub_name}"
    len_keys = set(ds["test"].features.keys()) & LEN_KEYS
    dev_key = "dev" if "dev" in ds else "validation"
    lens = {"train": [], dev_key: [], "test": []}

    for split in lens.keys():
        if split not in ds:
            continue
        for k in len_keys:
            if isinstance(ds[split][k][0], str):
                lens[split] += [len(x) for x in ds[split][k]]
            elif isinstance(ds[split][k][0], list):
                assert isinstance(ds[split][k][0][0], str), f"Too nested: {k}"
                lens[split] += [len(y) for x in ds[split][k] for y in x]
            else:
                raise ValueError(f"Unknown type {type(ds[split][k])}")

    avg_lens = {
        k: round(sum(lens[k]) / len(lens[k]), 1) if lens[k] else 0 for k in lens
    }
    return (
        len(lens["train"]),
        len(lens[dev_key]),
        len(lens["test"]),
        avg_lens["train"],
        avg_lens[dev_key],
        avg_lens["test"],
    )


# Select all tasks
for task in MTEB().tasks:
    print("Task: ", task)
    if "dataset" in task.metadata_dict:
        hub_name = hub_url = task.metadata_dict["dataset"]["path"]
        ds_stats = get_ds_stats(hub_name.split("/")[-1])
    elif "beir_name" in task.metadata_dict:
        hub_name = hub_url = "BeIR/" + task.metadata_dict.get("beir_name")
        ds_stats = get_ds_stats_beir("/".join(hub_name.split("/")[1:]))
        if "cqadupstack" in hub_name:
            hub_url = "BeIR/cqadupstack-qrels"
    TABLE_STRING += "\n" + ONE_LINE.format(
        f"[{task.metadata_dict['name']}]({task.metadata_dict['reference']})",
        f"[{hub_name}](https://huggingface.co/datasets/{hub_url})",
        task.metadata_dict["description"],
        task.metadata_dict["type"],
        task.metadata_dict["category"],
        len(task.metadata_dict["eval_langs"]),
        *ds_stats,
    )

with open("./mdtable.md", "w") as f:
    f.write(TABLE_STRING)

# Convert to latex
for line in TABLE_STRING.split("\n")[2:]:
    if line:
        cols = line.split(" | ")
        idx = cols[0].index("]")
        cols[0] = cols[0][3:idx]
        cols[-1] = cols[-1][:-1]
        out = " & ".join(cols[:1] + cols[3:]) + " \\\\"
        print(out)
