#!/usr/bin/env python3
"""Script to upload Common Voice raw files (tar and tsv) directly to HuggingFace Hub"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files, login
from huggingface_hub.utils import RepositoryNotFoundError
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload Common Voice raw files to HuggingFace Hub"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing audio/ and transcripts/ folders",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on HuggingFace Hub (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="HuggingFace API token with write access",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Specific languages to upload (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist in the repository",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed uploads (default: 3)",
    )
    return parser.parse_args()


def discover_files(
    base_dir: str, languages: list[str] | None = None
) -> list[tuple[str, str]]:
    """Discover all tar and tsv files in the base directory.

    Returns:
        List of tuples (local_path, repo_path)
    """
    base_path = Path(base_dir)
    files_to_upload = []

    # Define file patterns to search for
    patterns = ["**/*.tar", "**/*.tsv"]

    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            # Get relative path from base directory
            relative_path = file_path.relative_to(base_path)

            # Check if we should filter by language
            if languages:
                # Extract language from path (assuming structure like audio/ab/... or transcripts/ab/...)
                parts = relative_path.parts
                if len(parts) >= 2:
                    potential_lang = parts[1]
                    if potential_lang not in languages:
                        continue

            # Convert to string paths
            local_path = str(file_path)
            repo_path = str(relative_path).replace("\\", "/")  # Ensure forward slashes

            files_to_upload.append((local_path, repo_path))

    # Also include metadata files at the root
    metadata_files = [
        "n_shards.json",
        "languages.py",
        "release_stats.py",
        "common_voice_21_0.py",
    ]
    for metadata_file in metadata_files:
        metadata_path = base_path / metadata_file
        if metadata_path.exists():
            files_to_upload.append((str(metadata_path), metadata_file))

    return sorted(files_to_upload)


def get_existing_files(api: HfApi, repo_id: str) -> set:
    """Get list of files already in the repository."""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        return set(files)
    except RepositoryNotFoundError:
        return set()


def upload_file_with_retry(
    api: HfApi, local_path: str, repo_path: str, repo_id: str, max_retries: int = 3
) -> bool:
    """Upload a file with retry logic.

    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {str(e)}")
                return False
    return False


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def main():
    args = parse_args()

    print(f"Base directory: {args.base_dir}")
    print(f"Repository: {args.repo_id}")
    print(f"Dry run: {args.dry_run}")
    print(f"Resume: {args.resume}")

    # Login to HuggingFace
    if not args.dry_run:
        login(args.token)
        api = HfApi()

        # Create repository if it doesn't exist
        try:
            api.repo_info(repo_id=args.repo_id, repo_type="dataset")
            print(f"Repository {args.repo_id} already exists.")
        except RepositoryNotFoundError:
            print(f"Creating repository {args.repo_id}...")
            api.create_repo(
                repo_id=args.repo_id, repo_type="dataset", private=args.private
            )

        # Get existing files if resuming
        existing_files = get_existing_files(api, args.repo_id) if args.resume else set()
    else:
        existing_files = set()

    # Discover files to upload
    print("\nDiscovering files...")
    files_to_upload = discover_files(args.base_dir, args.languages)

    if not files_to_upload:
        print("No files found to upload!")
        return

    # Filter out existing files if resuming
    if args.resume and existing_files:
        original_count = len(files_to_upload)
        files_to_upload = [
            (local, repo)
            for local, repo in files_to_upload
            if repo not in existing_files
        ]
        skipped_count = original_count - len(files_to_upload)
        if skipped_count > 0:
            print(
                f"Skipping {skipped_count} files that already exist in the repository."
            )

    print(f"Found {len(files_to_upload)} files to upload.")

    # Calculate total size
    total_size = 0
    for local_path, _ in files_to_upload:
        total_size += os.path.getsize(local_path)

    print(f"Total size to upload: {format_size(total_size)}")

    if args.dry_run:
        print("\nDRY RUN - Files that would be uploaded:")
        for local_path, repo_path in files_to_upload[:20]:  # Show first 20 files
            size = os.path.getsize(local_path)
            print(f"  {repo_path} ({format_size(size)})")
        if len(files_to_upload) > 20:
            print(f"  ... and {len(files_to_upload) - 20} more files")
        return

    # Upload files
    print("\nUploading files...")
    successful = 0
    failed = 0

    with tqdm(total=len(files_to_upload), desc="Uploading") as pbar:
        for local_path, repo_path in files_to_upload:
            try:
                size = os.path.getsize(local_path)
                pbar.set_description(f"Uploading {repo_path} ({format_size(size)})")

                if upload_file_with_retry(
                    api, local_path, repo_path, args.repo_id, args.max_retries
                ):
                    successful += 1
                else:
                    failed += 1
                    print(f"\nFailed to upload: {repo_path}")

            except Exception as e:
                failed += 1
                print(f"\nError uploading {repo_path}: {str(e)}")

            pbar.update(1)

    # Summary
    print("\n" + "=" * 50)
    print("Upload Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(files_to_upload)}")

    if successful > 0:
        print(f"\nFiles uploaded to: https://huggingface.co/datasets/{args.repo_id}")

    if failed > 0:
        print(
            "\nSome files failed to upload. You can run the script again with --resume to retry."
        )


if __name__ == "__main__":
    main()
