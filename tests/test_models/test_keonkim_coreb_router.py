from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import mteb
from mteb.models.model_implementations.keonkim_coreb_router import (
    ROUTES,
    CoREBTaskTypeRouter,
)


def _metadata(task_type: str) -> SimpleNamespace:
    return SimpleNamespace(type=task_type)


def test_router_is_registered_and_loadable() -> None:
    name = "keonkim/coreb-task-type-router-f2llmv2-330m-c2llm-7b"
    meta = mteb.get_model_meta(name)
    router = mteb.get_model(
        name,
        revision=meta.revision,
        model_loader=lambda _: Mock(),
    )

    assert isinstance(router, CoREBTaskTypeRouter)
    assert router.mteb_model_meta.name == name


@pytest.mark.parametrize("task_type", ["Retrieval", "Reranking"])
def test_router_selects_model_by_coarse_task_type(task_type: str) -> None:
    child = Mock()
    child.encode.return_value = task_type
    loader = Mock(return_value=child)
    router = CoREBTaskTypeRouter("keonkim/router", model_loader=loader)

    result = router.encode(
        Mock(),
        task_metadata=_metadata(task_type),  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
    )

    assert result == task_type
    loader.assert_called_once_with(ROUTES[task_type])


def test_router_releases_previous_type() -> None:
    retrieval = Mock()
    reranking = Mock()
    children = iter([retrieval, reranking])
    router = CoREBTaskTypeRouter(
        "keonkim/router",
        model_loader=lambda _: next(children),
    )

    router._load_model("Retrieval")
    router._load_model("Reranking")

    assert router._models == {"Reranking": reranking}


def test_router_rejects_unsupported_task_type() -> None:
    router = CoREBTaskTypeRouter("keonkim/router", model_loader=Mock())

    with pytest.raises(ValueError, match="Unsupported task type"):
        router.encode(
            Mock(),
            task_metadata=_metadata("Classification"),  # type: ignore[arg-type]
            hf_split="test",
            hf_subset="default",
        )
