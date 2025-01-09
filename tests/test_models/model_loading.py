from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import scan_cache_dir

from mteb import get_model, get_model_meta
from mteb.models.overview import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)


def teardown_function():
    hf_cache_info = scan_cache_dir()
    all_revisions = []
    for repo in list(hf_cache_info.repos):
        for revision in list(repo.revisions):
            all_revisions.append(revision.commit_hash)

    delete_strategy = scan_cache_dir().delete_revisions(*all_revisions)
    print("Will free " + delete_strategy.expected_freed_size_str)
    delete_strategy.execute()


def get_model_below_n_param_threshold(model_name: str) -> str:
    """Test that we can get all models with a number of parameters below a threshold."""
    model_meta = get_model_meta(model_name=model_name)
    assert model_meta is not None
    if model_meta.n_parameters is not None:
        if model_meta.n_parameters >= 2e9:
            return "Over threshold. Not tested."
        elif "API" in model_meta.framework:
            try:
                m = get_model(model_name)
                if m is not None:
                    del m
                    return "None"
            except Exception as e:
                logging.warning(f"Failed to load model {model_name} with error {e}")
                return e.__str__()
        try:
            m = get_model(model_name)
            if m is not None:
                del m
                return "None"
        except Exception as e:
            logging.warning(f"Failed to load model {model_name} with error {e}")
            return e.__str__()
        finally:
            teardown_function()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omit_previous_success",
        action="store_true",
        default=False,
        help="Omit models that have been successfully loaded in the past",
    )
    parser.add_argument(
        "--run_missing",
        action="store_true",
        default=False,
        help="Run the missing models in the registry that are missing from existing results.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        nargs="+",
        default=None,
        help="Run the script for specific model names, e.g. model_1, model_2",
    )
    parser.add_argument(
        "--model_name_file",
        type=str,
        default=None,
        help="Filename containing space-separated model names to test.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    output_file = Path(__file__).parent / "model_load_failures.json"

    args = parse_args()

    # Load existing results if the file exists
    results = {}
    if output_file.exists():
        with output_file.open("r") as f:
            results = json.load(f)

    if args.model_name:
        all_model_names = args.model_name
    elif args.model_name_file:
        all_model_names = []
        if Path(args.model_name_file).exists():
            with open(args.model_name_file) as f:
                all_model_names = f.read().strip().split()
        else:
            logging.warning(
                f"Model name file {args.model_name_file} does not exist. Exiting."
            )
            exit(1)
    else:
        omit_keys = []
        if args.run_missing:
            omit_keys = list(results.keys())
        elif args.omit_previous_success:
            omit_keys = [k for k, v in results.items() if v == "None"]

        all_model_names = list(set(MODEL_REGISTRY.keys()) - set(omit_keys))

    for model_name in all_model_names:
        error_msg = get_model_below_n_param_threshold(model_name)
        results[model_name] = error_msg

        results = dict(sorted(results.items()))

        # Write the results to the file after each iteration
        with output_file.open("w") as f:
            json.dump(results, f, indent=4)
