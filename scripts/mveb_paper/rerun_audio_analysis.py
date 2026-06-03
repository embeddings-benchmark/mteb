"""Rerun Shriya's video-vs-audio-video analysis from the results cache.

Loads results through ``mteb.ResultCache`` (defaults to ``~/.cache/mteb`` and
pulls the remote ``embeddings-benchmark/results`` repo). Uses mteb's task
registry to discover v/va task pairs and cross-modal v2a/a2v tasks.

Outputs four tables matching tables/audio_*.tex:
  - per-dataset audio delta sorted within AV-grounded / V-grounded
  - per-paradigm audio delta (NEW)
  - per-model audio delta
  - cross-modal retrieval score per dataset/direction
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mteb  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent


# AV-grounded source datasets (Appendix A, tab:av-provenance).
AV_GROUNDED = {
    "mteb/AVE-Dataset",
    "mteb/AVMeme-Exam",
    "mteb/AVQA_val",
    "mteb/AV-SpeakerBench",
    "mteb/AudioCaps_AV",
    "mteb/Daily-Omni",
    "mteb/MELD",
    "mteb/MUSIC-AVQA_cls-preprocessed",
    "mteb/OmniVideoBench_subset",
    "mteb/PerceptionTest_val",
    "mteb/RAVDESS_AV",
    "mteb/Shot2Story20K_test",
    "mteb/VALOR-32K",
    "mteb/VGGSound",
    "mteb/VGGSound_AV_RETRIEVAL",
    "mteb/Video-MME_short",
    "mteb/WorldSense_1min",
    "mteb/YouCook2_val",
    "mteb/worldqa",
    "zachz/AV-SpeakerBench-PC",
    "zachz/AVE-Dataset-PC-V",
    "zachz/AVE-Dataset-PC-VA",
    "zachz/MELD-PC-V",
    "zachz/MELD-PC-VA",
    "zachz/MUSIC-AVQA-PC-V",
    "zachz/MUSIC-AVQA-PC-VA",
    "zachz/RAVDESS-AV-PC-V",
    "zachz/RAVDESS-AV-PC-VA",
}

V_TO_VA = {"v2c": "va2c", "v2t": "va2t", "vt2t": "vat2t"}


# Map MTEB model name → embedding paradigm (from generate_main_results_table.py).
PARADIGM_MAP = {
    # MLLM-based embedding
    "BidirLM/BidirLM-Omni-2.5B-Embedding": "MLLM-based embedding",
    "Haon-Chen/e5-omni-3B": "MLLM-based embedding",
    "Haon-Chen/e5-omni-7B": "MLLM-based embedding",
    "LCO-Embedding/LCO-Embedding-Omni-3B": "MLLM-based embedding",
    "LCO-Embedding/LCO-Embedding-Omni-7B": "MLLM-based embedding",
    "Tevatron/OmniEmbed-v0.1": "MLLM-based embedding",
    "nvidia/omni-embed-nemotron-3b": "MLLM-based embedding",
    "jinaai/jina-embeddings-v5-omni-nano": "MLLM-based embedding",
    "jinaai/jina-embeddings-v5-omni-small": "MLLM-based embedding",
    "Qwen/Qwen3-VL-Embedding-2B": "MLLM-based embedding",
    "Qwen/Qwen3-VL-Embedding-8B": "MLLM-based embedding",
    "zhibinlan/UME-R1-2B": "MLLM-based embedding",
    "zhibinlan/UME-R1-7B": "MLLM-based embedding",
    "VLM2Vec/VLM2Vec-V2.0": "MLLM-based embedding",
    # Generative MLLM (hidden-state pooled)
    "Qwen/Qwen2.5-Omni-3B": "Generative MLLM",
    "Qwen/Qwen2.5-Omni-7B": "Generative MLLM",
    # Multimodal binding
    "encord-team/ebind-audio-vision": "Multimodal binding",
    "encord-team/ebind-full": "Multimodal binding",
    "encord-team/ebind-points-vision": "Multimodal binding",
    # Audio-visual contrastive
    "facebook/pe-av-small": "Audio-visual contrastive",
    "facebook/pe-av-base": "Audio-visual contrastive",
    "facebook/pe-av-large": "Audio-visual contrastive",
    # Video-text contrastive (no audio path)
    "microsoft/xclip-base-patch16": "Video-text contrastive",
    "microsoft/xclip-base-patch32": "Video-text contrastive",
    "microsoft/xclip-large-patch14": "Video-text contrastive",
}


def load_score_table(cache: mteb.ResultCache, tasks) -> pd.DataFrame:
    """Per-task (rows) × per-model (cols) main_score table from the cache."""
    return (
        cache.load_results(tasks=tasks, require_model_meta=False)
        .to_dataframe()
        .set_index("task_name")
    )


def read_task_score(scores: pd.DataFrame, model_name: str, task_name: str) -> float | None:
    if task_name not in scores.index or model_name not in scores.columns:
        return None
    val = scores.at[task_name, model_name]
    return None if pd.isna(val) else float(val)


def find_pairs(tasks):
    index = {}
    for t in tasks:
        path = t.metadata.dataset.get("path", "")
        task_type = t.metadata.type
        cat = t.metadata.category
        index[(path, task_type, cat)] = t
    pairs = []
    for (path, task_type, cat), v in index.items():
        if cat not in V_TO_VA:
            continue
        va = index.get((path, task_type, V_TO_VA[cat]))
        if va is None:
            continue
        pairs.append((v, va))
    return pairs


# Models whose video-only scores are at near-random levels on most paired
# tasks; including them is misleading for an "audio's contribution to
# video understanding" analysis because they have no video understanding
# to begin with.
EXCLUDE_MODELS = {
    "jinaai/jina-embeddings-v5-omni-nano",
    "jinaai/jina-embeddings-v5-omni-small",
}


def main(cache: mteb.ResultCache, out_dir: Path):
    tasks = list(mteb.get_tasks(modalities=["video"], exclude_beta=False))
    pairs = find_pairs(tasks)
    print(f"Found {len(pairs)} paired (v, va) task groups\n")

    scores = load_score_table(cache, tasks)
    models = [c for c in scores.columns]

    # Iterate every model with results and compute deltas.
    rows = []
    for model_name in models:
        if model_name in EXCLUDE_MODELS:
            continue
        for v_task, va_task in pairs:
            v_score = read_task_score(scores, model_name, v_task.metadata.name)
            va_score = read_task_score(scores, model_name, va_task.metadata.name)
            if v_score is None or va_score is None:
                continue
            dataset_path = v_task.metadata.dataset.get("path", "")
            rows.append({
                "model": model_name,
                "paradigm": PARADIGM_MAP.get(model_name, "Other"),
                "dataset": dataset_path.split("/")[-1],
                "dataset_path": dataset_path,
                "task_type": v_task.metadata.type,
                "v_task": v_task.metadata.name,
                "va_task": va_task.metadata.name,
                "v_score": v_score,
                "va_score": va_score,
                "delta": va_score - v_score,
                "av_grounded": dataset_path in AV_GROUNDED,
            })

    if not rows:
        print("No paired (v, va) results found in results/results.")
        return
    df = pd.DataFrame(rows)

    # ---------- Summary by AV-grounded vs V-grounded ----------
    print("=" * 72)
    print("  Mean audio delta  (AV-grounded  vs  V-grounded)")
    print("=" * 72)
    grouped = df.groupby("av_grounded")["delta"].agg(["mean", "std", "count"])
    grouped.index = grouped.index.map({True: "AV-grounded", False: "V-grounded"})
    grouped.columns = ["mean_delta", "std", "N"]
    grouped.index.name = None
    print(grouped.to_string(float_format=lambda x: f"{x:+.4f}"))
    print()

    # ---------- Per-paradigm ----------
    print("=" * 72)
    print("  Per-paradigm mean audio delta  (audio-capable models only)")
    print("=" * 72)
    # Restrict to models that actually pair both v and va (have any audio capability)
    audio_capable = df["model"].unique()
    paradigm_summary = (
        df.groupby("paradigm")
        .agg(
            mean_delta=("delta", "mean"),
            std=("delta", "std"),
            n_models=("model", "nunique"),
            N=("delta", "count"),
            min_per_model=("delta", lambda x: x.groupby(df.loc[x.index, "model"]).mean().min()),
            max_per_model=("delta", lambda x: x.groupby(df.loc[x.index, "model"]).mean().max()),
        )
        .sort_values("mean_delta", ascending=False)
    )
    print(paradigm_summary.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) and abs(x) < 1 else f"{x}"))
    print(f"\nModels evaluated (n={len(audio_capable)}): {sorted(audio_capable)}")
    print()

    # ---------- Per-model ----------
    print("=" * 72)
    print("  Per-model mean audio delta")
    print("=" * 72)
    model_summary = (
        df.groupby("model")
        .agg(
            paradigm=("paradigm", "first"),
            mean_delta=("delta", "mean"),
            std=("delta", "std"),
            N=("delta", "count"),
        )
        .sort_values("mean_delta", ascending=False)
    )
    print(model_summary.to_string(float_format=lambda x: f"{x:+.4f}" if isinstance(x, float) else f"{x}"))
    print()

    # ---------- Per-dataset ----------
    print("=" * 72)
    print("  Per-dataset mean audio delta (sorted within AV-grounded / V-grounded)")
    print("=" * 72)
    ds_summary = (
        df.groupby(["dataset", "task_type", "av_grounded"])["delta"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta", "std": "std", "count": "N"})
        .reset_index()
    )
    ds_summary["group"] = ds_summary["av_grounded"].map({True: "AV-grounded", False: "V-grounded"})
    ds_summary = ds_summary.sort_values(["av_grounded", "mean_delta"], ascending=[False, False])
    print(ds_summary[["group", "dataset", "task_type", "mean_delta", "std", "N"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print()

    # ---------- Cross-modal v2a/a2v ----------
    print("=" * 72)
    print("  Cross-modal retrieval scores  (v2a / a2v)")
    print("=" * 72)
    cross_rows = []
    for t in tasks:
        if t.metadata.category not in ("v2a", "a2v"):
            continue
        for model_name in models:
            if model_name in EXCLUDE_MODELS:
                continue
            score = read_task_score(scores, model_name, t.metadata.name)
            if score is None:
                continue
            cross_rows.append({
                "dataset": t.metadata.dataset.get("path", "").split("/")[-1],
                "direction": t.metadata.category,
                "model": model_name,
                "score": score,
            })
    if cross_rows:
        cross_df = pd.DataFrame(cross_rows)
        cross_sum = (
            cross_df.groupby(["dataset", "direction"])["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values(["dataset", "direction"])
        )
        print(cross_sum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        cross_df = pd.DataFrame(columns=["dataset", "direction", "model", "score"])
        print("(none found)")

    emit_tables(df, cross_df, out_dir)


def _fmt_signed(x):
    return f"${x:+.3f}$"


def _fmt_unsigned(x):
    return f"${x:.3f}$"


def _ds_display(name: str) -> str:
    return {
        "AVE-Dataset": "AVE-Dataset",
        "AVMeme-Exam": "AVMeme-Exam",
        "AVQA_val": "AVQA",
        "AV-SpeakerBench": "AV-SpeakerBench",
        "AudioCaps_AV": "AudioCaps-AV",
        "Daily-Omni": "Daily-Omni",
        "MELD": "MELD",
        "MUSIC-AVQA_cls-preprocessed": "MUSIC-AVQA",
        "OmniVideoBench_subset": "OmniVideoBench",
        "PerceptionTest_val": "PerceptionTest",
        "RAVDESS_AV": "RAVDESS-AV",
        "Shot2Story20K_test": "Shot2Story20K",
        "VALOR-32K": "VALOR-32K",
        "VGGSound": "VGGSound",
        "VGGSound_AV_RETRIEVAL": "VGGSound-AV-Ret.",
        "Video-MME_short": "Video-MME",
        "WorldSense_1min": "WorldSense",
        "YouCook2_val": "YouCook2",
        "worldqa": "WorldQA",
        "ActivityNet_Captions_val2": "ActivityNet",
        "Breakfast": "Breakfast",
        "DiDeMo": "DiDeMo",
        "diving48v2": "Diving48",
        "EgoSchema_subset": "EgoSchema",
        "HMDB51": "HMDB51",
        "Human-Animal-Cartoon": "Human-Animal-Cartoon",
        "MSR-VTT": "MSR-VTT",
        "MSVD": "MSVD",
        "NExT-QA": "NExT-QA",
        "SomethingSomethingV2": "SomethingSomethingV2",
        "TUNA-Bench_1K": "TUNA-Bench",
        "UCF101-51VA": "UCF101-51VA",
        "VATEX_test_1k": "VATEX",
        "kinetics-400": "Kinetics-400",
        "kinetics-600": "Kinetics-600",
        "kinetics-700-2020": "Kinetics-700-2020",
        "panda-70m": "Panda-70M",
    }.get(name, name)


_TASK_TYPE_DISPLAY = {
    "Any2AnyRetrieval": "Retrieval",
    "VideoCentricQA": "QA",
    "VideoClassification": "Classification",
    "VideoClustering": "Clustering",
    "VideoPairClassification": "Pair Classification",
    "VideoZeroshotClassification": "ZS Classification",
}


def _model_short(name: str) -> str:
    return name.split("/")[-1]


def emit_tables(df, cross_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------ audio_delta_per_dataset.tex ------------
    ds = (
        df.groupby(["dataset", "task_type", "av_grounded"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_delta", "count": "N"})
    )
    ds["display"] = ds["dataset"].map(_ds_display)
    ds["tt_display"] = ds["task_type"].map(_TASK_TYPE_DISPLAY)
    av_rows = ds[ds["av_grounded"]].sort_values("mean_delta", ascending=False)
    v_rows = ds[~ds["av_grounded"]].sort_values("mean_delta", ascending=False)

    lines = [
        r"% AUTOGENERATED by scripts/mveb_paper/rerun_audio_analysis.py — do not edit by hand.",
        r"\begin{table*}[p]",
        r"\small",
        r"\centering",
        r"\caption{Per-dataset audio delta ($\Delta = \text{score}_{\text{va}} - \text{score}_{\text{v}}$), sorted by mean delta within each annotation group. AV-grounded datasets had labels produced from both audio and visual content; V-grounded datasets had labels produced from visuals alone, even though the source clips often carry audio (see \autoref{tab:av-provenance}). $N$ counts model--dataset pairs with results for both task variants.}",
        r"\label{tab:audio_per_dataset}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Task type & $\bar\Delta$ & $\sigma$ & $N$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{AV-grounded datasets}} \\",
        r"\midrule",
    ]
    for _, r in av_rows.iterrows():
        lines.append(
            f"{r['display']:25s} & {r['tt_display']:18s} & {_fmt_signed(r['mean_delta'])} & {_fmt_unsigned(r['std'])} & ${int(r['N'])}$ \\\\"
        )
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{V-grounded datasets}} \\",
        r"\midrule",
    ])
    for _, r in v_rows.iterrows():
        lines.append(
            f"{r['display']:25s} & {r['tt_display']:18s} & {_fmt_signed(r['mean_delta'])} & {_fmt_unsigned(r['std'])} & ${int(r['N'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (out_dir / "audio_delta_per_dataset.tex").write_text("\n".join(lines))

    # ------------ audio_per_model.tex ------------
    ms = (
        df.groupby("model")["delta"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_delta", "count": "N"})
        .sort_values("mean_delta", ascending=False)
        .reset_index()
    )
    n_models = df["model"].nunique()
    lines = [
        r"% AUTOGENERATED by scripts/mveb_paper/rerun_audio_analysis.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Per-model mean audio delta ($\bar\Delta$) across 48 paired task groups for the {n_models} audio-capable models in our roster. A positive $\bar\Delta$ indicates the model benefits on average from the audio track; a negative value indicates the audio channel hurts performance. $N$ counts dataset--task pairs evaluated.}}",
        r"\label{tab:audio_per_model}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & $\bar\Delta$ & $\sigma$ & $N$ \\",
        r"\midrule",
    ]
    for _, r in ms.iterrows():
        lines.append(
            f"{_model_short(r['model']):30s} & {_fmt_signed(r['mean_delta'])} & {_fmt_unsigned(r['std'])} & ${int(r['N'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "audio_per_model.tex").write_text("\n".join(lines))

    # ------------ audio_cross_modal.tex ------------
    cs = (
        cross_df.groupby(["dataset", "direction"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_score", "count": "N"})
        .sort_values(["dataset", "direction"])
    )
    cs["display"] = cs["dataset"].map(_ds_display)
    lines = [
        r"% AUTOGENERATED by scripts/mveb_paper/rerun_audio_analysis.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Cross-modal retrieval scores ($\bar{{s}}$) averaged across {n_models} models, for datasets that have v2a (video-to-audio) and a2v (audio-to-video) retrieval task variants. High scores indicate that video and audio embeddings are well-aligned, making explicit audio information more likely redundant. \textit{{YouCook2}} has the lowest cross-modal alignment and gains $\Delta = +0.056$ in the paired v/va comparison, consistent with spoken narration providing information not recoverable from visual frames alone. By contrast, \textit{{VGGSound-AV-Ret.}} has high alignment yet yields $\Delta = -0.035$, indicating the video embedding already captures the acoustic structure of these clips.}}",
        r"\label{tab:audio_cross_modal}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Dir. & $\bar{s}$ & $\sigma$ & $N$ \\",
        r"\midrule",
    ]
    for _, r in cs.iterrows():
        lines.append(
            f"{r['display']:18s} & {r['direction']} & {_fmt_unsigned(r['mean_score'])} & {_fmt_unsigned(r['std'])} & ${int(r['N'])}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out_dir / "audio_cross_modal.tex").write_text("\n".join(lines))

    print(f"\nWrote 3 .tex tables to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Path to a results cache directory (default: ~/.cache/mteb).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force-refresh the remote results repo before loading "
        "(otherwise the local cache is used, downloading only if absent).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory the .tex tables are written to (default: this script's "
        "folder). Point at the paper's tables/ directory to write there directly.",
    )
    args = parser.parse_args()
    cache = mteb.ResultCache(cache_path=args.cache_path)
    if args.download or not cache.has_remote:
        cache.download_from_remote()
    main(cache=cache, out_dir=Path(args.out_dir))
