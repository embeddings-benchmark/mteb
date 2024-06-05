import importlib.util

IMPORT_ERROR = "{0} requires the `{1}` library but it was not found in your environment. If you want to load {2} models, please `pip install {1}` else they will not be available."


def _is_package_available(pkg_name: str) -> bool:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    return package_exists


def requires_package(obj: object, package_name: str, model_name: str) -> None:
    if not _is_package_available(package_name):
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        raise ImportError(IMPORT_ERROR.format(name, package_name, model_name))
