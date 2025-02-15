#!/usr/bin/env python3
"""This script scans all Python files in the "models" directory, imports each module,
identifies variables that are ModelMeta instances, computes their memory_usage_mb via
the calculate_memory_usage_mb method, and then updates the source code in place by
inserting or replacing the "memory_usage_mb" keyword argument in the ModelMeta constructor.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re
from typing import Any

from tqdm import tqdm

# IMPORTANT: Adjust the import below to point to the module where ModelMeta is defined.
# For example, if ModelMeta is defined in a file model_meta.py in your package, do:
# from model_meta import ModelMeta
from mteb.model_meta import ModelMeta  # <-- Replace with the actual import path


def find_matching_paren(text: str, open_index: int) -> int | None:
    """Given text and the index of an opening parenthesis, return the index of the
    matching closing parenthesis.
    """
    count: int = 0
    for i in range(open_index, len(text)):
        if text[i] == "(":
            count += 1
        elif text[i] == ")":
            count -= 1
            if count == 0:
                return i
    return None


def find_modelmeta_call_range(text: str, var_name: str) -> tuple[int, int] | None:
    """Given the source text and a variable name, find the range (start, end)
    of the ModelMeta constructor call that assigns to that variable.
    This function uses a regex to locate the assignment and then uses a
    parenthesis matcher to capture the entire call.
    """
    pattern: str = rf"^{var_name}\s*=\s*ModelMeta\s*\("
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    start: int = match.start()
    # Locate the '(' after "ModelMeta"
    open_paren_index: int = text.find("(", match.end() - 1)
    if open_paren_index == -1:
        return None
    end_paren_index: int | None = find_matching_paren(text, open_paren_index)
    if end_paren_index is None:
        return None
    # Return the range covering the entire ModelMeta( ... ) call.
    return start, end_paren_index + 1


def update_memory_usage_in_call(call_text: str, memory_usage: float | None) -> str:
    """Update (or insert) the memory_usage_mb keyword argument in a ModelMeta(...) call.
    If memory_usage_mb exists, its value is updated.
    Otherwise, it is inserted right after the n_parameters argument.
    """
    mem_usage_str: str = str(memory_usage) if memory_usage is not None else "None"

    if "memory_usage_mb" in call_text:
        # Update existing memory_usage_mb using a lambda to avoid backreference issues.
        updated_call: str = re.sub(
            r"(memory_usage_mb\s*=\s*)([^,\)\n]+)",
            lambda m: m.group(1) + mem_usage_str,
            call_text,
        )
        return updated_call
    else:
        # Try to locate the n_parameters argument to insert after it.
        match = re.search(r"(n_parameters\s*=\s*[^,]+,\s*)", call_text)
        if match:
            insertion_point: int = match.end()
            new_param: str = f"memory_usage_mb={mem_usage_str}, "
            updated_call: str = (
                call_text[:insertion_point] + new_param + call_text[insertion_point:]
            )
            return updated_call
        else:
            # Fallback: if n_parameters is not found, insert before the closing parenthesis.
            stripped: str = call_text.rstrip()
            if not stripped.endswith(")"):
                return call_text
            return call_text[:-1] + f", memory_usage_mb={mem_usage_str}" + call_text[-1]


def update_file(file_path: str) -> None:
    """For a given Python file, import the module, iterate over its attributes to find
    ModelMeta instances, compute memory_usage_mb for each, and update the source
    code accordingly.
    """
    with open(file_path, encoding="utf-8") as f:
        content: str = f.read()

    # Import the module from the file
    module_name: str = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        print(f"Could not load module from {file_path}")
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error importing {file_path}: {e}")
        return

    # List of modifications to apply: each is a tuple (start, end, replacement_text)
    modifications: list[tuple[int, int, str]] = []
    for attr_name in tqdm(dir(module), desc=f"Processing {file_path}"):
        if attr_name.startswith("__"):
            continue
        obj: Any = getattr(module, attr_name)
        if isinstance(obj, ModelMeta):
            # Compute memory_usage_mb via the instance method.
            mem_usage: float | None = obj.calculate_memory_usage_mb()
            # Find the corresponding ModelMeta(...) call in the source file.
            call_range: tuple[int, int] | None = find_modelmeta_call_range(
                content, attr_name
            )
            if call_range is None:
                print(f"Could not find definition for {attr_name} in {file_path}")
                continue
            start, end = call_range
            original_call_text: str = content[start:end]
            if "memory_usage_mb" in original_call_text:
                continue
            updated_call_text: str = update_memory_usage_in_call(
                original_call_text, mem_usage
            )
            if original_call_text != updated_call_text:
                modifications.append((start, end, updated_call_text))
                print(
                    f"Updating {attr_name} in {file_path}: setting memory_usage_mb={mem_usage}"
                )

    # Apply modifications in reverse order to avoid shifting indices.
    if modifications:
        modifications.sort(key=lambda mod: mod[0], reverse=True)
        for start, end, replacement in modifications:
            content = content[:start] + replacement + content[end:]
        # Write the updated content back to the file.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(f"No modifications needed for {file_path}")


def main() -> None:
    """Main function: scans the "models" directory for .py files and updates each."""
    models_dir: str = (
        "../mteb/models"  # Change this if your models are in a different folder.
    )
    py_files: list[str] = glob.glob(os.path.join(models_dir, "*.py"))
    if not py_files:
        print(f"No Python files found in {models_dir}")
        return

    for file_path in py_files:
        update_file(file_path)


if __name__ == "__main__":
    main()
