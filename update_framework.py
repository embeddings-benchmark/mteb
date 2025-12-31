import ast
import csv
import logging
from pathlib import Path

from huggingface_hub import model_info
from tqdm.auto import tqdm

from mteb.models.get_model_meta import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Master framework list to match against HuggingFace tags
MASTER_FRAMEWORK_LIST = {
    "sentence-transformers",
    "transformers",
    "onnx",
    "safetensors",
    "gguf",
}

# Mapping from HuggingFace tags to framework names used in MTEB
TAG_TO_FRAMEWORK = {
    "sentence-transformers": "Sentence Transformers",
    "transformers": "Transformers",
    "onnx": "ONNX",
    "safetensors": "safetensors",
    "gguf": "GGUF",
}


def get_framework_from_hf_tags(model_name: str) -> list[str] | None:
    """Extract frameworks from HuggingFace model tags.

    Args:
        model_name: HuggingFace model name

    Returns:
        List of framework names found in tags, or None if model info couldn't be fetched
    """
    try:
        info = model_info(model_name)
        if not info.tags:
            return None

        frameworks = []
        for tag in info.tags:
            tag_lower = tag.lower()
            if tag_lower in MASTER_FRAMEWORK_LIST:
                framework_name = TAG_TO_FRAMEWORK.get(tag_lower)
                if framework_name and framework_name not in frameworks:
                    frameworks.append(framework_name)

        return frameworks if frameworks else None
    except Exception as e:
        logger.warning(f"Failed to fetch info for {model_name}: {e}")
        return None


def find_modelmeta_in_file(file_path: Path, model_name: str) -> tuple[int, int] | None:
    """Find the line range of a ModelMeta definition for a given model name in a file.

    Args:
        file_path: Path to the Python file
        model_name: The model name to search for

    Returns:
        Tuple of (start_line, end_line) in the file, or None if not found
    """
    try:
        content = file_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if (
                        isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "ModelMeta"
                    ):
                        # Check if this ModelMeta has the right name
                        for kw in node.value.keywords:
                            if kw.arg == "name":
                                try:
                                    if (
                                        isinstance(kw.value, ast.Constant)
                                        and kw.value.value == model_name
                                    ):
                                        return (node.lineno - 1, node.end_lineno)
                                except (AttributeError, TypeError):
                                    pass
    except Exception as e:
        logger.warning(f"Error parsing {file_path}: {e}")

    return None


def update_framework_in_file(
    file_path: Path, start_line: int, end_line: int, new_frameworks: list[str]
) -> bool:
    """Update the framework field in a ModelMeta definition in a file.

    Args:
        file_path: Path to the Python file
        start_line: Start line of the ModelMeta definition (0-indexed)
        end_line: End line of the ModelMeta definition (1-indexed)
        new_frameworks: New list of frameworks to set

    Returns:
        True if update was successful, False otherwise
    """
    try:
        lines = file_path.read_text().splitlines(keepends=True)
        modelmeta_text = "".join(lines[start_line:end_line])

        # Find and update the framework field
        updated_text = modelmeta_text

        # Create the new framework list as Python code
        framework_str = "[" + ", ".join(f'"{fw}"' for fw in new_frameworks) + "]"

        # Replace framework field using regex
        import re

        pattern = r"framework\s*=\s*\[[^\]]*\]"
        updated_text = re.sub(pattern, f"framework={framework_str}", updated_text)

        # Write back to file
        lines[start_line:end_line] = [updated_text]
        file_path.write_text("".join(lines))
        return True
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False


def find_model_implementation_files() -> dict[str, Path]:
    """Find all model implementation files and return mapping of model_name -> file_path.

    Returns:
        Dictionary mapping model names to their implementation file paths
    """
    model_to_file = {}
    impl_dir = Path(__file__).parent / "mteb" / "models" / "model_implementations"

    if not impl_dir.exists():
        logger.error(f"Model implementations directory not found: {impl_dir}")
        return model_to_file

    for py_file in impl_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue

        try:
            content = py_file.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    if isinstance(node.value, ast.Call):
                        if (
                            isinstance(node.value.func, ast.Name)
                            and node.value.func.id == "ModelMeta"
                        ):
                            for kw in node.value.keywords:
                                if kw.arg == "name":
                                    try:
                                        if isinstance(kw.value, ast.Constant):
                                            model_name = kw.value.value
                                            model_to_file[model_name] = py_file
                                    except (AttributeError, TypeError):
                                        pass
        except Exception as e:
            logger.warning(f"Error parsing {py_file}: {e}")

    return model_to_file


def save_results_to_csv(
    results: dict, output_file: str = "framework_update_results.csv"
) -> None:
    """Save results to a CSV file for easy verification.

    Args:
        results: Dictionary containing update results
        output_file: Path to the output CSV file
    """
    csv_path = Path(__file__).parent / output_file

    with Path.open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(
            ["Status", "Model Name", "File Path", "Old Frameworks", "New Frameworks"]
        )

        # Write updated models
        for item in sorted(results["updated_models"], key=lambda x: x["name"]):
            old_frameworks = ", ".join(item["old"]) if item["old"] else "None"
            new_frameworks = ", ".join(item["new"])
            writer.writerow(
                ["UPDATED", item["name"], item["file"], old_frameworks, new_frameworks]
            )

        # Write unchanged models
        for name in sorted(results["unchanged_models"]):
            writer.writerow(["UNCHANGED", name, "", "", ""])

        # Write not found models
        for name in sorted(results["not_found_models"]):
            writer.writerow(["NOT FOUND", name, "", "", ""])

        # Write failed models
        for name in sorted(results["failed_models"]):
            writer.writerow(["FAILED", name, "", "", ""])

    print(f"\n✅ Results saved to: {csv_path.absolute()}")


def update_framework_fields():
    """Update framework fields for all models in registry based on HuggingFace tags."""
    models_updated = []
    models_failed = []
    models_unchanged = []
    models_not_found = []

    # Get mapping of model names to their implementation files
    model_to_file = find_model_implementation_files()

    for model_name, meta in tqdm(
        MODEL_REGISTRY.items(),
        desc="Updating framework fields from HuggingFace tags",
        total=len(MODEL_REGISTRY),
    ):
        # Skip models without a name or with None loader
        if not meta.name or meta.loader is None:
            models_unchanged.append(meta.name or model_name)
            continue

        # Get frameworks from HuggingFace tags
        hf_frameworks = get_framework_from_hf_tags(meta.name)

        if hf_frameworks is None:
            # If we couldn't fetch from HF, keep existing framework
            if meta.framework:
                models_unchanged.append(meta.name)
            else:
                models_failed.append(meta.name)
            continue

        # Merge existing frameworks with new ones from HF (retain existing, add new)
        merged_frameworks = list(meta.framework) if meta.framework else []
        new_frameworks_added = False

        for fw in hf_frameworks:
            if fw not in merged_frameworks:
                merged_frameworks.append(fw)
                new_frameworks_added = True

        # Update only if there are new frameworks to add
        if new_frameworks_added:
            old_framework = list(meta.framework) if meta.framework else []

            # Update the ModelMeta object in memory
            meta.framework = merged_frameworks

            # Find and update the source file
            if meta.name in model_to_file:
                file_path = model_to_file[meta.name]
                line_range = find_modelmeta_in_file(file_path, meta.name)

                if line_range:
                    start_line, end_line = line_range
                    if update_framework_in_file(
                        file_path, start_line, end_line, merged_frameworks
                    ):
                        models_updated.append(
                            {
                                "name": meta.name,
                                "old": old_framework,
                                "new": merged_frameworks,
                                "file": str(file_path),
                            }
                        )
                    else:
                        models_failed.append(meta.name)
                else:
                    models_not_found.append(meta.name)
            else:
                models_not_found.append(meta.name)
        else:
            models_unchanged.append(meta.name)

    # Print summary to terminal
    print("\n" + "=" * 80)
    print("FRAMEWORK UPDATE SUMMARY")
    print("=" * 80)

    print(f"\n✅ Models updated: {len(models_updated)}")
    print(f"⏭️  Models unchanged: {len(models_unchanged)}")
    print(f"⚠️  Models with implementation not found: {len(models_not_found)}")
    print(f"❌ Models failed/missing: {len(models_failed)}")

    print("\n" + "=" * 80)

    results = {
        "updated": len(models_updated),
        "unchanged": len(models_unchanged),
        "failed": len(models_failed),
        "not_found": len(models_not_found),
        "updated_models": models_updated,
        "unchanged_models": models_unchanged,
        "failed_models": models_failed,
        "not_found_models": models_not_found,
    }

    # Save results to CSV
    save_results_to_csv(results)

    total = (
        results["updated"]
        + results["unchanged"]
        + results["failed"]
        + results["not_found"]
    )
    print(f"\nTotal models processed: {total}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_framework_fields()
