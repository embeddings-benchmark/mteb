from ebr.core.base import RetrievalDataset
from ebr.core.meta import DatasetMeta, dataset_id
from ebr.datasets.text import *
from ebr.utils.lazy_import import LazyImport


DATASET_REGISTRY: dict[str, DatasetMeta] = {}
for name in dir():
    meta = eval(name)
    # Explicitly exclude `LazyImport` instances since the latter check invokes the import.
    if not isinstance(meta, LazyImport) and isinstance(meta, DatasetMeta):
        DATASET_REGISTRY[meta._id] = eval(name)


def get_retrieval_dataset(
    data_path: str,
    dataset_name: str,
    **kwargs
) -> RetrievalDataset:
    key = dataset_id(dataset_name)
    return DATASET_REGISTRY[key].load_dataset(data_path=data_path, **kwargs)
