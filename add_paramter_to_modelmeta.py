import ast
import csv
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from mteb.models.get_model_meta import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_parameters_from_csv(csv_path: Path) -> dict[str, dict[str, int | None]]:
    """Load model parameters from CSV file."""
    parameters = {}

    with Path.open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]

            def parse(x):
                if not x:
                    return None
                return int(x.replace(",", "_").strip())

            parameters[model_name] = {
                "active_parameters": parse(row.get("active_params")),
                "embedding_parameters": parse(row.get("input_embedding_params")),
            }

    return parameters


def find_modelmeta_call_by_name(
    content: str, model_name: str
) -> tuple[int, int] | None:

    """Locate the exact ModelMeta(name="...") call in the file.
    Returns character offsets (start, end).
    """
    tree = ast.parse(content)
    lines = content.split("\n")

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "ModelMeta":
                for kw in node.keywords:
                    if (
                        kw.arg == "name"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == model_name
                    ):
                        start = sum(len(l) + 1 for l in lines[: node.lineno - 1])
                        end = sum(len(l) + 1 for l in lines[: node.end_lineno])
                        return start, end

    return None


def find_model_implementation_files() -> dict[str, Path]:
    """Map model_name -> source file where ModelMeta(name="...") is defined."""
    model_to_file = {}
    impl_dir = Path(__file__).parent / "mteb" / "models" / "model_implementations"

    for py_file in impl_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "ModelMeta":
                        for kw in node.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                model_to_file[kw.value.value] = py_file
        except Exception as e:
            logger.warning(f"Failed parsing {py_file}: {e}")

    return model_to_file


def update_modelmeta_parameters(
    file_path: Path,
    model_name: str,
    active_params: int | None,
    embedding_params: int | None,
) -> bool:

    """Insert active_parameters and embedding_parameters
    immediately after n_parameters inside the ModelMeta(...) call.
    """
    content = file_path.read_text(encoding="utf-8")
    loc = find_modelmeta_call_by_name(content, model_name)

    if not loc:
        logger.warning(f"ModelMeta not found for {model_name}")
        return False

    start, end = loc
    block = content[start:end]

    # Idempotency: do not reinsert
    if "active_parameters" in block or "embedding_parameters" in block:
        return False

    idx = block.find("n_parameters")
    if idx == -1:
        logger.warning(f"n_parameters not found for {model_name}")
        return False

    comma = block.find(",", idx)
    if comma == -1:
        return False

    def fmt(x):
        return "None" if x is None else f"{x:_}"

    insertion = (
        f",\n    active_parameters={fmt(active_params)}"
        f",\n    embedding_parameters={fmt(embedding_params)}"
    )

    updated_block = block[:comma] + insertion + block[comma:]
    updated_content = content[:start] + updated_block + content[end:]

    file_path.write_text(updated_content, encoding="utf-8")
    return True


def main():
    """Update ModelMeta definitions with active_parameters and embedding_parameters from CSV."""
    csv_path = Path("model total and active parameters - model_parameters.csv")
    result_log = Path("parameter_update_results.csv")

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return

    params = load_parameters_from_csv(csv_path)
    model_to_file = find_model_implementation_files()

    results = []

    for model_name in tqdm(sorted(MODEL_REGISTRY.keys()), desc="Updating ModelMeta"):
        p = params.get(model_name, {})
        active = p.get("active_parameters")
        embed = p.get("embedding_parameters")

        if model_name not in model_to_file:
            results.append(
                {
                    "model_name": model_name,
                    "status": "not_found_in_impl",
                }
            )
            continue

        file_path = model_to_file[model_name]

        ok = update_modelmeta_parameters(file_path, model_name, active, embed)

        results.append(
            {
                "model_name": model_name,
                "status": "updated" if ok else "unchanged_or_failed",
                "file": file_path.name,
            }
        )

        pd.DataFrame(results).to_csv(result_log, index=False)

    logger.info(f"Finished. Results saved to {result_log}")


if __name__ == "__main__":
    main()
