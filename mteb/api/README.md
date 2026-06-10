# mteb.api

FastAPI surface that powers the leaderboardv2 SvelteKit frontend.

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

| Method | Path                           | Returns                                                                                   |
|--------|--------------------------------|-------------------------------------------------------------------------------------------|
| GET    | `/health`                      | `{"ok": true}`                                                                            |
| GET    | `/v1/benchmarks/menu`          | Nested menu tree (matches frontend `MenuEntry[]`).                                        |
| GET    | `/v1/benchmarks`               | Flat list of leaderboard benchmarks.                                                      |
| GET    | `/v1/benchmarks/{name}`        | Single benchmark metadata.                                                                |
| GET    | `/v1/benchmarks/{name}/scores` | Full summary with rows, per-task scores, per-task-type means. (Legacy alias: `/summary`.) |

JSON keys are emitted in `camelCase` to match the frontend types in
`leaderboardv2/src/lib/types.ts`.

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
