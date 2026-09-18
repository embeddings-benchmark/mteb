import os
import sys
import zipfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio


def process_acm_dataset(
    hf_token: str,
    source_repo: str = "chuonghm/ACM",
    target_repo: str = "deep9539/ACM-processed",
    work_dir: str = "./acm_workspace",
    keep_extracted_media: bool = True,
):
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    media_dir = work_path / "extracted_media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Skip download & extraction if media is already unpacked
    extracted_files = list(media_dir.glob("*"))
    if media_dir.exists() and len(extracted_files) > 0:
        print(
            f"[✓] Extracted media directory '{media_dir}' exists with {len(extracted_files)} items. Skipping download & extraction."
        )
    else:
        print(f"[*] Downloading media.zip from {source_repo}...")
        zip_path = hf_hub_download(
            repo_id=source_repo,
            filename="media.zip",
            repo_type="dataset",
            token=hf_token,
            local_dir=work_path,
        )

        print("[*] Unpacking media archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(media_dir)
        print(f"[✓] Unpacked media files to {media_dir}")

        if os.path.exists(zip_path):
            os.remove(zip_path)

    # -------------------------------------------------------------
    # Subset 1: Process Candidates (audio_path -> audio)
    # -------------------------------------------------------------
    print("\n[*] Processing subset: composed_audio_retrieval_candidates...")
    try:
        try:
            ds_cand = load_dataset(
                source_repo, "composed_audio_retrieval_candidates", token=hf_token
            )
        except Exception:
            ds_cand = load_dataset(
                source_repo,
                data_dir="data/composed_audio_retrieval_candidates",
                token=hf_token,
            )

        def resolve_candidate_audio(example):
            rel_path = example.get("audio_path")
            if rel_path:
                full_path = str(media_dir / rel_path)
                example["audio"] = full_path if os.path.exists(full_path) else None
            return example

        print("[*] Linking candidate audio files...")
        ds_cand = ds_cand.map(resolve_candidate_audio)
        ds_cand = ds_cand.cast_column("audio", Audio(sampling_rate=None))

        print(f"[*] Uploading candidate dataset to {target_repo}...")
        ds_cand.push_to_hub(
            repo_id=target_repo,
            config_name="composed_audio_retrieval_candidates",
            token=hf_token,
            private=False,
        )
        print("[✓] Candidates subset processed & uploaded successfully.")

    except Exception as e:
        print(f"[!] Error in candidates subset: {e}", file=sys.stderr)

    # -------------------------------------------------------------
    # Subset 2: Process Queries (src_audio_path -> src_audio, tgt_audio_path -> tgt_audio)
    # -------------------------------------------------------------
    print("\n[*] Processing subset: composed_audio_retrieval_queries...")
    try:
        try:
            ds_queries = load_dataset(
                source_repo, "composed_audio_retrieval_queries", token=hf_token
            )
        except Exception:
            ds_queries = load_dataset(
                source_repo,
                data_dir="data/composed_audio_retrieval_queries",
                token=hf_token,
            )

        def resolve_query_audios(example):
            # Resolve Source Audio
            src_rel = example.get("src_audio_path")
            if src_rel:
                full_src = str(media_dir / src_rel)
                example["src_audio"] = full_src if os.path.exists(full_src) else None

            # Resolve Target Audio
            tgt_rel = example.get("tgt_audio_path")
            if tgt_rel:
                full_tgt = str(media_dir / tgt_rel)
                example["tgt_audio"] = full_tgt if os.path.exists(full_tgt) else None

            return example

        print("[*] Linking query source and target audio files...")
        ds_queries = ds_queries.map(resolve_query_audios)

        # Cast both audio columns to native Audio features
        ds_queries = ds_queries.cast_column("src_audio", Audio(sampling_rate=None))
        ds_queries = ds_queries.cast_column("tgt_audio", Audio(sampling_rate=None))

        print(f"[*] Uploading query dataset to {target_repo}...")
        ds_queries.push_to_hub(
            repo_id=target_repo,
            config_name="composed_audio_retrieval_queries",
            token=hf_token,
            private=False,
        )
        print("[✓] Queries subset processed & uploaded successfully.")

    except Exception as e:
        print(f"[!] Error in queries subset: {e}", file=sys.stderr)

    if not keep_extracted_media:
        shutil.rmtree(work_path, ignore_errors=True)


if __name__ == "__main__":
    token = os.getenv("HF_TOKEN") or input("Enter HF Token: ").strip()
    process_acm_dataset(hf_token=token)
