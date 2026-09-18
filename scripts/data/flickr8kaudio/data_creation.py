"""
Build deep9539/flickr-audio-image on the HF Hub from two Kaggle sources:

  1. adityajn105/flickr8k                     -> images/     + captions.txt
  2. warcoder/flickr-8k-audio-caption-corpus   -> wavs/       + wav2capt.txt

Produces three configs pushed to the same HF dataset repo:

  audio  : {id, audio}
  images : {id, image, text}          (one row per image; caption = first/joined caption)
  qrels  : {image_id, audio_id}       (relevance pairs)

Requirements:
    pip install kagglehub datasets huggingface_hub pillow soundfile pandas

Kaggle auth:
    Either have ~/.kaggle/kaggle.json in place, or set KAGGLE_USERNAME / KAGGLE_KEY
    env vars before running (kagglehub picks these up automatically).

HF auth:
    huggingface-cli login
    (or export HF_TOKEN=... before running)

Usage:
    python build_flickr_audio_image.py --push --repo_id deep9539/flickr-audio-image
    python build_flickr_audio_image.py            # dry run, no push, just builds locally
"""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def ffmpeg_reencode(
    src: Path, dst: Path, sample_rate: int = 16000
) -> tuple[Path, str | None]:
    """
    Re-encode one wav through ffmpeg to force clean, accurate headers/metadata
    (mono, fixed sample rate, PCM16). This is what fixes torchcodec's
    seek_mode="approximate" mis-seeking on files with imprecise duration/
    header metadata. Returns (dst_path, error_message_or_None).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return dst, result.stderr.strip()
        return dst, None
    except Exception as e:
        return dst, str(e)


def reencode_all(
    wav_paths: list[Path],
    out_dir: Path,
    sample_rate: int = 16000,
    max_workers: int = 16,
) -> dict:
    """
    Re-encode a list of wav files in parallel via ffmpeg into out_dir.
    Returns a dict mapping original stem -> new fixed path, and prints/collects failures.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. a static or shared build) "
            "and make sure it's exported before running this script."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem_to_fixed_path = {}
    failures = []

    print(f"Re-encoding {len(wav_paths)} wav files via ffmpeg -> {out_dir} ...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(ffmpeg_reencode, src, out_dir / src.name, sample_rate): src
            for src in wav_paths
        }
        done = 0
        for fut in as_completed(futures):
            src = futures[fut]
            dst, err = fut.result()
            if err:
                failures.append((src, err))
            else:
                stem_to_fixed_path[src.stem] = dst
            done += 1
            if done % 2000 == 0 or done == len(wav_paths):
                print(
                    f"  {done}/{len(wav_paths)} processed ({len(failures)} failures so far)"
                )

    if failures:
        print(f"\n{len(failures)} files failed to re-encode:")
        for src, err in failures[:20]:
            print(f"  {src}: {err}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    return stem_to_fixed_path


def download_sources() -> tuple[Path, Path]:
    """Download both Kaggle datasets via kagglehub, return local paths."""
    import kagglehub

    print("Downloading adityajn105/flickr8k (images + captions.txt) ...")
    images_root = Path(kagglehub.dataset_download("adityajn105/flickr8k"))

    print(
        "Downloading warcoder/flickr-8k-audio-caption-corpus (wavs + wav2capt.txt) ..."
    )
    audio_root = Path(
        kagglehub.dataset_download("warcoder/flickr-8k-audio-caption-corpus")
    )

    return images_root, audio_root


def find_file(root: Path, name: str) -> Path:
    """Recursively find a file by name under root (kagglehub nests dirs unpredictably)."""
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find '{name}' under {root}")
    return matches[0]


def find_dir(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find directory '{name}' under {root}")
    return matches[0]


def load_captions(images_root: Path) -> pd.DataFrame:
    """captions.txt format: image,caption  (header row: image,caption)"""
    captions_path = find_file(images_root, "captions.txt")
    df = pd.read_csv(captions_path)
    df.columns = [c.strip().lower() for c in df.columns]
    # expect columns: image, caption
    assert "image" in df.columns and "caption" in df.columns, (
        f"unexpected columns: {df.columns}"
    )
    return df


def load_wav2capt(audio_root: Path) -> pd.DataFrame:
    """
    wav2capt.txt format (whitespace separated, no header):
        2571096893_694ce79768_1.wav 2571096893_694ce79768.jpg #1
    caption number is 0-indexed (#0, #1, ...) matching order of captions
    for that image in Flickr8k.token.txt / captions.txt
    """
    wav2capt_path = find_file(audio_root, "wav2capt.txt")
    rows = []
    with open(wav2capt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            wav_name, jpg_name, capt_idx = parts
            capt_idx = int(capt_idx.lstrip("#"))
            rows.append({"wav": wav_name, "image": jpg_name, "caption_idx": capt_idx})
    return pd.DataFrame(rows)


def build_tables(
    images_root: Path, audio_root: Path, fixed_wav_dir: Path, sample_rate: int = 16000
):
    captions_df = load_captions(images_root)
    wav2capt_df = load_wav2capt(audio_root)

    images_dir = find_dir(images_root, "Images")
    wavs_dir = find_dir(audio_root, "wavs")

    # --- images config: one row per unique image, caption = first caption for that image ---
    captions_df["cap_order"] = captions_df.groupby("image").cumcount()
    first_caps = (
        captions_df[captions_df["cap_order"] == 0][["image", "caption"]]
        .rename(columns={"caption": "text"})
        .reset_index(drop=True)
    )
    first_caps["id"] = first_caps["image"].apply(lambda x: Path(x).stem)
    first_caps["image_path"] = first_caps["image"].apply(lambda x: str(images_dir / x))
    images_table = first_caps[["id", "image_path", "text"]].rename(
        columns={"image_path": "image"}
    )

    # --- audio config: one row per wav file, re-encoded through ffmpeg first ---
    wav2capt_df["id"] = wav2capt_df["wav"].apply(lambda x: Path(x).stem)
    original_paths = wav2capt_df["wav"].apply(lambda x: wavs_dir / x).tolist()

    stem_to_fixed = reencode_all(original_paths, fixed_wav_dir, sample_rate=sample_rate)

    # drop any rows whose file failed to re-encode, rather than silently pushing bad data
    before = len(wav2capt_df)
    wav2capt_df = wav2capt_df[wav2capt_df["id"].isin(stem_to_fixed.keys())].reset_index(
        drop=True
    )
    dropped = before - len(wav2capt_df)
    if dropped:
        print(f"Dropping {dropped} rows whose audio failed ffmpeg re-encoding.")

    wav2capt_df["audio_path"] = wav2capt_df["id"].apply(lambda s: str(stem_to_fixed[s]))
    audio_table = wav2capt_df[["id", "audio_path"]].rename(
        columns={"audio_path": "audio"}
    )

    # --- qrels: audio_id -> image_id it was spoken for ---
    wav2capt_df["image_id"] = wav2capt_df["image"].apply(lambda x: Path(x).stem)
    qrels_table = wav2capt_df[["image_id", "id"]].rename(columns={"id": "audio_id"})

    return audio_table, images_table, qrels_table


def to_hf_datasets(
    audio_table: pd.DataFrame, images_table: pd.DataFrame, qrels_table: pd.DataFrame
):
    from datasets import Dataset, Audio, Image

    audio_ds = Dataset.from_pandas(audio_table, preserve_index=False)
    audio_ds = audio_ds.cast_column("audio", Audio())

    images_ds = Dataset.from_pandas(images_table, preserve_index=False)
    images_ds = images_ds.cast_column("image", Image())

    qrels_ds = Dataset.from_pandas(qrels_table, preserve_index=False)

    return audio_ds, images_ds, qrels_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="deep9539/flickr-audio-image")
    ap.add_argument("--push", action="store_true", help="Actually push to the HF Hub")
    ap.add_argument("--private", action="store_true", help="Push as a private dataset")
    ap.add_argument(
        "--fixed_wav_dir",
        default=None,
        help="Where to write ffmpeg-re-encoded wavs. Defaults to ~/flickr_fixed_wavs",
    )
    ap.add_argument("--sample_rate", type=int, default=16000)
    args = ap.parse_args()

    fixed_wav_dir = (
        Path(args.fixed_wav_dir)
        if args.fixed_wav_dir
        else Path.home() / "flickr_fixed_wavs"
    )

    images_root, audio_root = download_sources()

    print("Building tables (re-encoding audio through ffmpeg for clean headers) ...")
    audio_table, images_table, qrels_table = build_tables(
        images_root,
        audio_root,
        fixed_wav_dir=fixed_wav_dir,
        sample_rate=args.sample_rate,
    )
    print(f"  audio:  {len(audio_table)} rows")
    print(f"  images: {len(images_table)} rows")
    print(f"  qrels:  {len(qrels_table)} rows")

    print("Converting to HF Datasets (this decodes/validates audio & image files) ...")
    audio_ds, images_ds, qrels_ds = to_hf_datasets(
        audio_table, images_table, qrels_table
    )

    # Sanity check a few rows before pushing anything
    print("Sanity-checking a few rows ...")
    for i in (0, len(audio_ds) - 1):
        a = audio_ds[i]["audio"]
        print(
            f"  audio[{i}] id={audio_ds[i]['id']} sr={a['sampling_rate']} n_samples={len(a['array'])}"
        )
    for i in (0, len(images_ds) - 1):
        img = images_ds[i]["image"]
        print(f"  images[{i}] id={images_ds[i]['id']} size={img.size} mode={img.mode}")

    if not args.push:
        print(
            "\nDry run only (pass --push to actually upload). Local objects built successfully."
        )
        return

    print(f"\nPushing to {args.repo_id} ...")
    # Smaller shards reduce the chance of binary-offset corruption on large audio pushes
    audio_ds.push_to_hub(
        args.repo_id, config_name="audio", private=args.private, max_shard_size="200MB"
    )
    images_ds.push_to_hub(
        args.repo_id, config_name="images", private=args.private, max_shard_size="200MB"
    )
    qrels_ds.push_to_hub(
        args.repo_id, config_name="qrels", private=args.private, max_shard_size="200MB"
    )
    print("Push complete. Running post-push verification (re-downloads from hub) ...")

    verify_push(args.repo_id)


def verify_push(repo_id: str):
    """Re-download the pushed audio config and confirm every row decodes cleanly."""
    from datasets import load_dataset

    verify_ds = load_dataset(
        repo_id, "audio", split="train", download_mode="force_redownload"
    )
    ids_only = verify_ds.select_columns(["id"])

    bad_rows = []
    for i in range(len(verify_ds)):
        try:
            verify_ds[i]["audio"].get_all_samples()
        except Exception as e:
            bad_rows.append((i, ids_only[i]["id"], repr(e)))

    total = len(verify_ds)
    print(
        f"\nVerification: {len(bad_rows)} / {total} rows failed to decode after push."
    )
    if bad_rows:
        print("First 20 failures:")
        for row in bad_rows[:20]:
            print(" ", row)
    else:
        print("All rows verified OK.")


if __name__ == "__main__":
    main()
