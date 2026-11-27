"""Compare four query inputs on the official and shared Expresso galleries.

With MTEB's audio dependencies installed, run:

    python scripts/data/inspire_expresso/diagnose_instructions.py order
    python scripts/data/inspire_expresso/diagnose_instructions.py clap --device cpu
    python scripts/data/inspire_expresso/diagnose_instructions.py iris --device mps

IRIS uses the public checkpoint's inference code and needs transformers==5.8.1,
accelerate==1.14.0 and soxr==1.0.0. The reference run used PyTorch 2.11,
bfloat16 and one example per forward pass. HF_HOME controls the download cache.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.tasks.retrieval.eng.inspire_expresso_retrieval import (
    INSPIREExpressoAT2ARetrieval,
)

IRIS = "lca0503/IRIS-Embed-Qwen2.5-Omni-3B"
IRIS_REVISION = "d364daf2a4fea4326c3d3e9dbbaf849499cabebf"
SPEAKER = "same speaker"
STYLE = "same speaking style"
BOTH = "same speaker and same speaking style"
GALLERY_DIGEST = "a0a67473915f22eafec63b3712e1a685b0fbf060853849bf11e125d334c33e01"


def load_shard(kind, columns):
    source = INSPIREExpressoAT2ARetrieval.metadata.dataset
    path = hf_hub_download(
        source["path"],
        f"{kind}/Expresso-00000-of-00001.parquet",
        repo_type="dataset",
        revision=source["revision"],
    )
    return pq.read_table(path, columns=columns)


def galleries(rows, documents):
    """Reconstruct reciprocal swaps and verify every shared candidate and positive."""
    lookup = {doc: i for i, doc in enumerate(documents)}
    gold = np.array([[lookup[d] for d in row["positive_documents"]] for row in rows])
    allowed = np.ones((len(rows), len(documents)), dtype=bool)
    references = defaultdict(dict)
    for i, row in enumerate(rows):
        intent = row["relevance"]
        family = intent if intent in (SPEAKER, STYLE, BOTH) else "named"
        references[row["audio_id"]][family] = i
        allowed[i, [lookup[d] for d in row["excluded_ids"]]] = False
    swap = np.empty(len(rows), dtype=int)
    pairs = []
    digest = hashlib.sha256()
    for family in references.values():
        for a, b in ((family[SPEAKER], family[STYLE]), (family[BOTH], family["named"])):
            swap[a], swap[b] = b, a
            shared = allowed[a] & allowed[b]
            positives = np.concatenate((gold[a], gold[b]))
            if len(set(positives)) == 4 and shared[positives].all():
                pairs.append((a, b, shared))
                record = [
                    f"q{a:05d}",
                    f"q{b:05d}",
                    rows[a]["positive_documents"],
                    rows[b]["positive_documents"],
                    documents[shared].tolist(),
                ]
                digest.update(
                    json.dumps(record, separators=(",", ":")).encode() + b"\n"
                )
    if len(pairs) != 283 or digest.hexdigest() != GALLERY_DIGEST:
        raise ValueError("Shared galleries differ from the pinned Expresso pilot.")
    return gold, allowed, swap, pairs


def unit(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def quietness_order(audio, tie_order):
    """Rank by waveform RMS, without using a query or any source metadata."""
    from scipy.io import wavfile

    powers = []
    for clip in audio:
        _, pcm = wavfile.read(io.BytesIO(clip.as_buffer()))
        powers.append(np.mean(np.square(pcm, dtype=np.float64)))
    return tie_order[np.argsort(np.asarray(powers)[tie_order], kind="stable")]


def registered_order_scores():
    """Exercise the evaluator through its normal search protocol without reading inputs."""

    class PositionSearch:
        mteb_model_meta = None

        def index(self, corpus, **kwargs):
            self.scores = {doc: float(i) for i, doc in enumerate(corpus["id"])}

        def search(self, queries, *, top_k, top_ranked, **kwargs):
            if top_ranked is not None or top_k != len(self.scores):
                raise ValueError("The search model must receive the full gallery.")
            return dict.fromkeys(queries["id"], self.scores)

    scores = INSPIREExpressoAT2ARetrieval().evaluate(
        PositionSearch(), encode_kwargs={}
    )["default"]
    return {key: scores[key] for key in ("recall_at_1", "ndcg_at_10")}


def measure(scores, cases, gold, tie_order):
    """Rank only eligible recordings, breaking ties by descending content ID."""
    ranks = []
    for i, mask in cases:
        candidates = tie_order[mask[tie_order]]
        values = scores[i, candidates]
        ranking = candidates[np.argsort(-values, kind="stable")]
        ranks.append(np.flatnonzero(np.isin(ranking, gold[i])) + 1)
    ranks = np.asarray(ranks)
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        "ndcg_at_10": float(
            np.mean(np.sum((ranks <= 10) / np.log2(1 + ranks), axis=1))
            / (1 + 1 / np.log2(3))
        ),
    }, ranks[:, 0]


def score_vectors(banks, panels, tie_order):
    corpus, audio, text, joint = banks
    gold, allowed, swap, pairs = panels
    corpus = unit(corpus).T
    common = [(i, mask) for a, b, mask in pairs for i in (a, b)]
    output = {"shared_pairs": len(pairs), "official": {}, "paired": {}}
    first = {}
    joint_scores = unit(joint) @ corpus
    for mode, scores in (
        ("audio_only", unit(audio) @ corpus),
        ("text_only", unit(text) @ corpus),
        ("correct", joint_scores),
        ("swapped", joint_scores[swap]),
    ):
        output["official"][mode], _ = measure(
            scores, enumerate(allowed), gold, tie_order
        )
        output["paired"][mode], first[mode] = measure(scores, common, gold, tie_order)
    improvement = (first["correct"] < first["swapped"]).reshape(-1, 2)
    output["pairs_both_improved"] = int(np.sum(improvement.all(axis=1)))
    return output


def clap_vectors(query, document, rows, device):
    from mteb.models.model_implementations.clap_models import (
        ClapZeroShotWrapper,
        clap_htsat_fused,
    )

    checkpoint = snapshot_download(
        clap_htsat_fused.name,
        revision=clap_htsat_fused.revision,
        allow_patterns=["*.json", "merges.txt", "pytorch_model.bin"],
    )
    model = ClapZeroShotWrapper(
        checkpoint, revision=clap_htsat_fused.revision, device=device
    )
    model.model.eval()

    def audio(table):
        return model.get_audio_embeddings(
            DataLoader(Dataset(table.select(["audio"])), batch_size=8)
        )

    _, first, inverse = np.unique(
        [r["audio_id"] for r in rows], return_index=True, return_inverse=True
    )
    queries = audio(query.take(first))[inverse]
    texts = Dataset.from_dict({"text": query["instruction"].to_pylist()})
    text = model.get_text_embeddings(DataLoader(texts, batch_size=8))
    return audio(document), queries, text, queries + text


def iris_vectors(query, document, rows, device):
    import soxr
    from scipy.io import wavfile
    from transformers import AutoModelForCausalLM, AutoProcessor

    checkpoint = snapshot_download(IRIS, revision=IRIS_REVISION)
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()

    def encode(mode, text, audio_bytes=None):
        prefix = "Document: " if mode == "document" else f"Instruction: {text}\nQuery: "
        content = [{"type": "text", "text": prefix}]
        kwargs = {"text_kwargs": {"padding": True}, "return_tensors": "pt"}
        if audio_bytes is not None:
            sr, pcm = wavfile.read(io.BytesIO(audio_bytes))
            wave = soxr.resample(
                pcm.astype(np.float32) / 32768, sr, 16000, quality="HQ"
            )
            length = math.ceil(len(pcm) * 16000 / sr)
            wave = np.pad(wave, (0, max(0, length - len(wave))))[:length]
            content.append({"type": "audio"})
            kwargs.update(
                audio=[wave],
                audio_weight_text=[text] if mode == "correct" else None,
                audio_kwargs={
                    "sampling_rate": 16000,
                    "max_length": math.ceil(length / 160) * 160 + 640,
                },
            )
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=[prompt], **kwargs)
        if mode == "audio_only":
            router = processor.instruction_tokenizer(
                ["Instruction: \nQuery:"], return_tensors="pt"
            )
            inputs["audio_weight_input_ids"] = router["input_ids"]
            inputs["audio_weight_attention_mask"] = router["attention_mask"]
        output = model(
            **inputs.to(device),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = output.hidden_states[-1][0, -1].float()
        return torch.nn.functional.normalize(hidden, dim=0).cpu().numpy()

    q_audio = pc.struct_field(query["audio"], "bytes")
    d_audio = pc.struct_field(document["audio"], "bytes")
    audio, text, joint = {}, {}, []
    for i, row in enumerate(tqdm(rows, desc="IRIS queries")):
        ref, instruction = row["audio_id"], row["instruction"]
        clip = q_audio[i].as_buffer()
        if ref not in audio:
            audio[ref] = encode("audio_only", "", clip)
        if instruction not in text:
            text[instruction] = encode("text_only", instruction)
        joint.append(encode("correct", instruction, clip))
    docs = [
        encode("document", "", clip.as_buffer())
        for clip in tqdm(d_audio, desc="IRIS documents")
    ]
    return (
        docs,
        [audio[r["audio_id"]] for r in rows],
        [text[r["instruction"]] for r in rows],
        joint,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["order", "clap", "iris"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    np.random.seed(42)
    query = load_shard(
        "query",
        [
            "audio",
            "audio_id",
            "instruction",
            "relevance",
            "positive_documents",
            "excluded_ids",
        ],
    )
    document = load_shard("document", ["audio", "audio_id"])
    rows = query.drop(["audio"]).to_pylist()
    references = {}
    for row, clip in zip(rows, pc.struct_field(query["audio"], "bytes"), strict=True):
        digest = hashlib.sha256(clip.as_buffer()).digest()
        if references.setdefault(row["audio_id"], digest) != digest:
            raise ValueError("Repeated reference IDs must contain identical audio.")
    documents = np.array(document["audio_id"].to_pylist())
    panels = galleries(rows, documents)
    hashes = [
        hashlib.sha256(clip.as_buffer()).hexdigest()
        for clip in pc.struct_field(document["audio"], "bytes")
    ]
    tie_order = np.argsort(hashes)[::-1]
    if args.model == "order":
        gold, allowed, _, _ = panels
        scores = np.zeros(allowed.shape[1], dtype=np.float32)
        matrix = np.broadcast_to(scores, allowed.shape)
        result = {
            name: measure(matrix, enumerate(allowed), gold, order)[0]
            for name, order in (
                ("release_id_descending", np.argsort(documents)[::-1]),
                ("content_id_descending", tie_order),
                (
                    "quietest_document_first",
                    quietness_order(
                        pc.struct_field(document["audio"], "bytes"), tie_order
                    ),
                ),
            )
        }
        result["registered_row_position"] = registered_order_scores()
    else:
        encoder = iris_vectors if args.model == "iris" else clap_vectors
        result = score_vectors(
            encoder(query, document, rows, args.device), panels, tie_order
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
