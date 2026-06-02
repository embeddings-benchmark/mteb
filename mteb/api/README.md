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

First call to `/benchmarks/{name}/summary` downloads the results repo to
`~/.cache/mteb` (slow once, fast forever after). Set `MTEB_API_PRELOAD=1` to
warm the cache on startup in a background thread.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"ok": true}` |
| GET | `/benchmarks/menu` | Nested menu tree (matches frontend `MenuEntry[]`). |
| GET | `/benchmarks` | Flat list of leaderboard benchmarks. |
| GET | `/benchmarks/{name}` | Single benchmark metadata. |
| GET | `/benchmarks/{name}/summary` | Full summary with rows, per-task scores, per-task-type means. |

JSON keys are emitted in `camelCase` to match the frontend types in
`leaderboardv2/src/lib/types.ts`.

## CORS

Defaults allow `localhost:5173` (vite dev) and `localhost:4173` (vite preview).
Add more origins with `MTEB_API_CORS_ORIGINS=https://a.com,https://b.com`.
