import importlib
from pathlib import Path

from mteb.models.model_meta import ModelMeta


def get_all_model_meta_objects():
    """Import all ModelMeta objects from all modules in the implementations package."""
    model_meta_objects = []

    # Get the directory containing this __init__.py file
    package_dir = Path(__file__).parent

    # Iterate through all Python files in the directory
    for file_path in package_dir.glob("*.py"):
        # Skip __init__.py
        if file_path.name == "__init__.py":
            continue

        # Get module name
        module_name = file_path.stem

        # Import the module
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Get all objects from the module that are instances of ModelMeta
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # Check if it's an instance of ModelMeta
            if isinstance(attr, ModelMeta):
                model_meta_objects.append(attr)

    return model_meta_objects


# build a registry of all model meta objects
MODEL_REGISTRY = {meta.name: meta for meta in get_all_model_meta_objects()}
