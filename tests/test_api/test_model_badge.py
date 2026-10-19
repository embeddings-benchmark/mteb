"""Tests for the shields.io badge endpoint `/v1/models/{name}/badge`."""

from __future__ import annotations

from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mteb.api import aggregators, cache, routes
from mteb.api.schemas import (
    BenchmarkSummarySchema,
    ModelMetaSchema,
    SummaryRowSchema,
)

BENCHMARK = "MTEB(eng, v2)"
MODEL = "mteb/baseline-random-encoder"
OTHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _row(name: str, rank: int, mean_task: float | None, mean_task_type: float | None):
    return SummaryRowSchema.model_construct(
        rank=rank,
        model=ModelMetaSchema.model_construct(name=name),
        mean_task=mean_task,
        mean_task_type=mean_task_type,
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    rows = [
        _row(OTHER_MODEL, 1, 0.6512, 0.6),
        _row(MODEL, 2, None, 0.31234),
    ]

    async def fake_get_summary(name: str) -> BenchmarkSummarySchema:
        return BenchmarkSummarySchema.model_construct(benchmark_name=name, rows=rows)

    monkeypatch.setattr(cache, "get_summary", fake_get_summary)
    monkeypatch.setattr(aggregators, "_summary_row_indices", {})
    monkeypatch.setattr(routes, "_PROCESS_START_DATE", "2026-10-19")

    app = FastAPI()
    app.include_router(routes.router, prefix="/v1")
    return TestClient(app)


def _badge_url(model: str, benchmark: str) -> str:
    return f"/v1/models/{model}/badge?benchmark={quote(benchmark)}"


def test_badge_has_shields_endpoint_format(client: TestClient):
    response = client.get(_badge_url(OTHER_MODEL, BENCHMARK))

    assert response.status_code == 200
    assert response.json() == {
        "schemaVersion": 1,
        "label": BENCHMARK,
        "message": "#1 / 2 · 65.12 · 2026-10-19",
        "color": "2563eb",
        "labelColor": "1d1f26",
        "cacheSeconds": 4 * 60 * 60,
    }


def test_badge_falls_back_to_mean_task_type(client: TestClient):
    response = client.get(_badge_url(MODEL, BENCHMARK))

    assert response.status_code == 200
    assert response.json()["message"] == "#2 / 2 · 31.23 · 2026-10-19"


@pytest.mark.parametrize(
    ("model", "benchmark", "detail"),
    [
        ("org/not-a-model", BENCHMARK, "Unknown model"),
        (MODEL, "NotABenchmark", "Unknown benchmark"),
        ("intfloat/multilingual-e5-small", BENCHMARK, "has no results on benchmark"),
    ],
)
def test_badge_404(client: TestClient, model: str, benchmark: str, detail: str):
    response = client.get(_badge_url(model, benchmark))

    assert response.status_code == 404
    assert detail in response.json()["detail"]


def test_badge_route_does_not_shadow_model_detail(client: TestClient):
    response = client.get(f"/v1/models/{MODEL}")

    assert response.status_code == 200
    assert response.json()["name"] == MODEL
