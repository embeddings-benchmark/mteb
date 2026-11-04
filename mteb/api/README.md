# mteb.api

FastAPI surface that powers the leaderboard frontend.

## Install

```sh
pip install -e ".[api]"
```

## Run

```sh
uvicorn mteb.api.app:app --reload --port 8000
```

First call to `/v1/benchmarks/{name}/scores` downloads the results repo to
`~/.cache/mteb` (slow once, fast forever after). Set `PRELOAD=1` to
warm the cache on startup in a background thread.

## Endpoints

Data routes are mounted under `/v1/`; infra routes (`/health`, `/metrics`,
`/robots.txt`, `/favicon.ico`, `/og/*`) stay at the root.

| Method | Path                                 | Returns                                                                                   |
|--------|--------------------------------------|-------------------------------------------------------------------------------------------|
| GET    | `/health`                            | `{"ok": true}`                                                                            |
| GET    | `/v1/benchmarks/menu`                | Nested menu tree (matches frontend `MenuEntry[]`).                                        |
| GET    | `/v1/benchmarks`                     | Flat list of leaderboard benchmarks.                                                      |
| GET    | `/v1/benchmarks/{name}`              | Single benchmark metadata.                                                                |
| GET    | `/v1/benchmarks/{name}/scores`       | Full summary with rows, per-task scores, per-task-type means. (Legacy alias: `/summary`.) |
| GET    | `/v1/tasks/{name}/descriptive_statistics` | Per-split descriptive statistics for the task (raw JSON, structure varies by task type).  |
| GET    | `/v1/models/{name}/badge?benchmark=…` | [shields.io endpoint badge](https://shields.io/badges/endpoint-badge) with the model's rank, score and data date on a benchmark. |

JSON keys are emitted in `camelCase` to match the frontend types in
`leaderboardv2/src/lib/types.ts`.

## Badges

`/v1/models/{name}/badge` returns a shields.io endpoint badge, so a model card
or README can show its current leaderboard position:

```markdown
![MTEB](https://img.shields.io/endpoint?url=https%3A%2F%2Fmteb-leaderboard-backend.hf.space%2Fv1%2Fmodels%2Fintfloat%2Fmultilingual-e5-small%2Fbadge%3Fbenchmark%3DMTEB%28eng%2C%2520v2%29)
```

The message reads `#rank / total · score · date`. The score is `Mean (Task)`
(falling back to `Mean (TaskType)`) on the 0-100 scale, and the date is when
the API loaded the leaderboard data. The inner URL must be URL-encoded. A model
without results on the benchmark returns 404.

## CORS

Defaults allow all origins (`*`) — this is a public read-only service.
Restrict with `CORS_ORIGINS=https://a.com,https://b.com`.

## Observability

Per-route Prometheus metrics are exposed at `/metrics` (counter, latency
histogram, in-flight gauge, exceptions). The `handler` label is the matched
route template, so cardinality stays bounded.

OpenTelemetry tracing ships via OTLP HTTP — one span per request, with
W3C `traceparent` context extracted from inbound headers. Metrics and
logs are deliberately *not* wired through OTEL (Prometheus already
handles metrics; stdlib logging is left alone). The pipeline is a no-op
unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set, so local dev and the test
suite don't try to push to a collector that isn't there.

| Env var                       | Purpose                                                                            |
|-------------------------------|------------------------------------------------------------------------------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector base URL, e.g. `http://localhost:4318`. Setting this turns telemetry on. |
| `OTEL_SERVICE_NAME`           | Service name tagged on every signal. Defaults to `mteb-api`.                       |
| `OTEL_RESOURCE_ATTRIBUTES`    | Extra `key=value,key=value` pairs merged into the resource.                        |
| `OTEL_EXPORTER_OTLP_HEADERS`  | Auth headers for hosted collectors (Honeycomb, Grafana Cloud, etc.).               |
