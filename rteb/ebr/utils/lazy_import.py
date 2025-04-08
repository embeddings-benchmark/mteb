import importlib
import importlib.util
from types import ModuleType
from typing import Any, Optional


def prompt_install(
    package: str,
    version: Optional[str] = None
) -> bool:
    """Checks whether the user wants to install a module before proceeding.
    """
    raise ModuleNotFoundError(f"{package}{'==' + version if version else ''} not found.")


class LazyImport(ModuleType):
    """Lazily import a module to avoid unnecessary dependencies. If a required
    dependency does not exist, it will prompt the user for it.

    Adapted from fzliu/radient/utils/lazy_loader.py.
    """

    def __init__(
        self,
        name: str,
        attribute: Optional[str] = None,
        package_name: Optional[str] = None,
        min_version: Optional[str] = None
    ):
        super().__init__(name)
        self._attribute = attribute
        self._top_name = name.split(".")[0]
        self._package_name = package_name if package_name else self._top_name
        self._min_version = min_version
        self._module = None

    def __call__(self, *args, **kwargs) -> Any:
        return self._evaluate()(*args, **kwargs)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._evaluate(), attribute)

    def __dir__(self) -> list:
        return dir(self._evaluate())

    def _evaluate(self) -> ModuleType:
        if not self._module:
            if not importlib.util.find_spec(self._top_name):
                prompt_install(self._package_name, self._min_version)
        self._module = importlib.import_module(self.__name__)
        if self._min_version and self._module.__version__ < self._min_version:
            prompt_install(self._package_name, self._min_version)
            self._module = importlib.import_module(self.__name__)
        if self._attribute:
            return getattr(self._module, self._attribute)
        return self._module