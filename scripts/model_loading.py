from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from mteb import get_model, get_model_meta
from mteb.models.overview import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)


CACHE_FOLDER = Path(__file__).parent / ".cache"


def teardown_function():
    """Remove cache folder and its contents"""
    for item in CACHE_FOLDER.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


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
                    return "None"
            except Exception as e:
                logging.warning(f"Failed to load model {model_name} with error {e}")
                return e.__str__()
        try:
            m = get_model(model_name, cache_folder=CACHE_FOLDER)
            if m is not None:
                return "None"
        except TypeError:  # when cache_folder is not supported.
            try:
                m = get_model(model_name)
                if m is not None:
                    return "None"
            except Exception as e:
                logging.warning(f"Failed to load model {model_name} with error {e}")
                return e.__str__()
            finally:
                teardown_function()
        except Exception as e:
            logging.warning(f"Failed to load model {model_name} with error {e}")
            return e.__str__()
        finally:
            teardown_function()


if __name__ == "__main__":
    output_file = Path(__file__).parent / "failures.json"

    # Load existing results if the file exists
    results = {}
    if output_file.exists():
        with output_file.open("r") as f:
            results = json.load(f)

    all_model_names = list(set(MODEL_REGISTRY.keys()) - set(results.keys()))
    for model_name in all_model_names:
        error_msg = get_model_below_n_param_threshold(model_name)
        results[model_name] = error_msg

        results = dict(sorted(results.items()))

        # Write the results to the file after each iteration
        with output_file.open("w") as f:
            json.dump(results, f, indent=4)
