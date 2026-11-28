


The following is a AI-generated comparison.

Personally after reviewing it I think the scores better reflects reality.

1) models that drop are generally quite small (performs disproportionally well) and models that are bumped are typically quite popular. 
2) There are a few changes in the pareto status' of models. Mostly in the european case where there is a less stable ranking (due to fewer people running on it)

This file should be deleted before merge

-----

# Ranking changes: MTEB(Multilingual, v2→v3) and MTEB(Europe, v1→v2)

Removed tasks: TempReasonL1, SpartQA, WinoGrande, WebLINXCandidatesReranking, plus KorSarcasmClassification (Multilingual only). See embeddings-benchmark/mteb#5416.
Results: embeddings-benchmark/results @ d49c00d25 (2026-10-16). Both versions are scored with the leaderboard summary-table code on the same loaded results, so only the task set differs.

## MTEB(Multilingual, v2) → MTEB(Multilingual, v3)

Models ranked: 451 (models with results on all tasks get a Mean(Task); Borda covers all models, as on the leaderboard).

- Borda rank: Kendall τ = 0.995, Spearman ρ = 1.000
- Mean(Task) rank (the 87 models complete on both): Kendall τ = 0.991
- Models that only become complete (get a Mean(Task)) in the new version, because they were missing just removed tasks: 0
- Models whose Borda rank changes: 173; median |Δ| = 0, max |Δ| = 26
- Top-10 by Borda (old): microsoft/harrier-oss-v1-27b, tencent/KaLM-Embedding-Gemma3-12B-2511, nvidia/llama-embed-nemotron-8b, Qwen/Qwen3-Embedding-8B, google/gemini-embedding-001, Qwen/Qwen3-Embedding-4B, Octen/Octen-Embedding-8B, codefuse-ai/F2LLM-v2-14B, codefuse-ai/F2LLM-v2-8B, microsoft/harrier-oss-v1-0.6b
- Top-10 by Borda (new): microsoft/harrier-oss-v1-27b, tencent/KaLM-Embedding-Gemma3-12B-2511, nvidia/llama-embed-nemotron-8b, Qwen/Qwen3-Embedding-8B, google/gemini-embedding-001, Qwen/Qwen3-Embedding-4B, Octen/Octen-Embedding-8B, codefuse-ai/F2LLM-v2-14B, codefuse-ai/F2LLM-v2-8B, Bytedance/Seed1.6-embedding-1215

### Top 50 (by either version), sorted by new Borda rank

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| microsoft/harrier-oss-v1-27b | 1 → 1 | +0 | 74.27 → 74.74 | 1 → 1 | +0 |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 2 → 2 | +0 | 72.32 → 73.10 | 2 → 2 | +0 |
| nvidia/llama-embed-nemotron-8b | 3 → 3 | +0 | 69.46 → 71.00 | 5 → 6 | -1 |
| Qwen/Qwen3-Embedding-8B | 4 → 4 | +0 | 70.58 → 72.23 | 3 → 3 | +0 |
| google/gemini-embedding-001 | 5 → 5 | +0 | 68.37 → 69.93 | 9 → 8 | +1 |
| Qwen/Qwen3-Embedding-4B | 6 → 6 | +0 | 69.45 → 71.05 | 6 → 5 | +1 |
| Octen/Octen-Embedding-8B | 7 → 7 | +0 | 67.84 → 69.33 | 11 → 11 | +0 |
| codefuse-ai/F2LLM-v2-14B | 8 → 8 | +0 | 68.74 → 70.09 | 8 → 7 | +1 |
| codefuse-ai/F2LLM-v2-8B | 9 → 9 | +0 | 68.09 → 69.41 | 10 → 10 | +0 |
| Bytedance/Seed1.6-embedding-1215 | 11 → 10 | +1 | 70.26 → 71.67 | 4 → 4 | +0 |
| microsoft/harrier-oss-v1-0.6b | 10 → 11 | -1 | 69.01 → 69.52 | 7 → 9 | -2 |
| codefuse-ai/F2LLM-v2-4B | 12 → 12 | +0 | 67.06 → 68.45 | 12 → 13 | -1 |
| ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826 | 13 → 13 | +0 | 65.64 → 67.08 | 15 → 17 | -2 |
| jinaai/jina-embeddings-v5-omni-small | 14 → 14 | +0 | 67.00 → 68.57 | 13 → 12 | +1 |
| jinaai/jina-embeddings-v5-text-small | 15 → 15 | +0 | – → – | – → – | – |
| codefuse-ai/F2LLM-v2-1.7B | 16 → 16 | +0 | 65.21 → 66.68 | 18 → 18 | +0 |
| BidirLM/BidirLM-Omni-2.5B-Embedding | 17 → 17 | +0 | 63.50 → 64.96 | 24 → 23 | +1 |
| ai-sage/Giga-Embeddings-instruct-3B-0826 | 18 → 18 | +0 | 63.89 → 65.43 | 20 → 20 | +0 |
| Qwen/Qwen3-Embedding-0.6B | 20 → 19 | +1 | 64.34 → 65.86 | 19 → 19 | +0 |
| jinaai/jina-embeddings-v5-omni-nano | 21 → 20 | +1 | 65.52 → 67.13 | 16 → 15 | +1 |
| jinaai/jina-embeddings-v5-text-nano | 21 → 20 | +1 | 65.52 → 67.13 | 16 → 15 | +1 |
| microsoft/harrier-oss-v1-270m | 19 → 22 | -3 | 66.55 → 67.40 | 14 → 14 | +0 |
| BidirLM/BidirLM-1.7B-Embedding | 24 → 23 | +1 | 63.54 → 65.07 | 22 → 22 | +0 |
| ICT-TIME-and-Querit/ICT-TIME-and-Querit-embedding-v1 | 26 → 24 | +2 | 63.78 → 65.24 | 21 → 21 | +0 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 23 → 25 | -2 | 62.51 → 63.72 | 28 → 29 | -1 |
| Linq-AI-Research/Linq-Embed-Mistral | 25 → 26 | -1 | – → – | – → – | – |
| intfloat/multilingual-e5-large-instruct | 27 → 27 | +0 | 63.22 → 64.65 | 25 → 26 | -1 |
| BidirLM/BidirLM-1B-Embedding | 28 → 28 | +0 | 63.21 → 64.81 | 26 → 25 | +1 |
| ICT-TIME-and-Querit/BOOM_4B_v1 | 29 → 29 | +0 | 63.52 → 64.95 | 23 → 24 | -1 |
| codefuse-ai/F2LLM-v2-0.6B | 30 → 30 | +0 | 62.74 → 64.19 | 27 → 27 | +0 |
| google/embeddinggemma-300m | 31 → 31 | +0 | 61.15 → 62.47 | 30 → 30 | +0 |
| Salesforce/SFR-Embedding-Mistral | 32 → 32 | +0 | – → – | – → – | – |
| google/text-multilingual-embedding-002 | 33 → 33 | +0 | 62.16 → 63.74 | 29 → 28 | +1 |
| GritLM/GritLM-7B | 34 → 34 | +0 | 60.92 → 62.22 | 31 → 32 | -1 |
| Cohere/Cohere-embed-multilingual-v3.0 | 37 → 35 | +2 | – → – | – → – | – |
| intfloat/e5-mistral-7b-instruct | 36 → 36 | +0 | 60.25 → 61.69 | 35 → 35 | +0 |
| GritLM/GritLM-8x7B | 35 → 37 | -2 | 60.49 → 61.94 | 33 → 33 | +0 |
| Lajavaness/bilingual-embedding-large | 39 → 38 | +1 | – → – | – → – | – |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 38 → 39 | -1 | – → – | – → – | – |
| BidirLM/BidirLM-0.6B-Embedding | 40 → 40 | +0 | 60.42 → 61.89 | 34 → 34 | +0 |
| Salesforce/SFR-Embedding-2_R | 41 → 41 | +0 | 59.80 → 61.30 | 36 → 36 | +0 |
| codefuse-ai/F2LLM-v2-330M | 42 → 42 | +0 | 60.84 → 62.33 | 32 → 31 | +1 |
| openai/text-embedding-3-large | 43 → 43 | +0 | – → – | – → – | – |
| NovaSearch/stella_en_1.5B_v5 | 44 → 44 | +0 | 56.53 → 57.86 | 47 → 46 | +1 |
| NovaSearch/jasper_en_vision_language_v1 | 45 → 45 | +0 | – → – | – → – | – |
| jinaai/jina-embeddings-v3 | 50 → 46 | +4 | 58.37 → 60.00 | 38 → 37 | +1 |
| OrdalieTech/Solon-embeddings-large-0.1 | 47 → 47 | +0 | – → – | – → – | – |
| nvidia/NV-Embed-v2 | 46 → 48 | -2 | – → – | – → – | – |
| BAAI/bge-m3 | 49 → 49 | +0 | – → – | – → – | – |
| mongodb/voyage-3.5 | 48 → 50 | -2 | 58.46 → 59.81 | 37 → 38 | -1 |

### Largest Mean(Task) rank movers (models complete on both versions)

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| ai-forever/ru-en-RoSBERTa | 152 → 153 | -1 | 41.57 → 42.31 | 76 → 78 | -2 |
| microsoft/harrier-oss-v1-0.6b | 10 → 11 | -1 | 69.01 → 69.52 | 7 → 9 | -2 |
| sdadas/mmlw-roberta-base | 128 → 127 | +1 | 41.51 → 42.55 | 78 → 76 | +2 |
| ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826 | 13 → 13 | +0 | 65.64 → 67.08 | 15 → 17 | -2 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 23 → 25 | -2 | 62.51 → 63.72 | 28 → 29 | -1 |
| intfloat/multilingual-e5-large-instruct | 27 → 27 | +0 | 63.22 → 64.65 | 25 → 26 | -1 |
| BidirLM/BidirLM-1B-Embedding | 28 → 28 | +0 | 63.21 → 64.81 | 26 → 25 | +1 |
| ICT-TIME-and-Querit/BOOM_4B_v1 | 29 → 29 | +0 | 63.52 → 64.95 | 23 → 24 | -1 |
| zeroentropy/zembed-1 | 64 → 65 | -1 | 56.58 → 57.78 | 46 → 47 | -1 |
| jinaai/jina-embeddings-v5-text-nano | 21 → 20 | +1 | 65.52 → 67.13 | 16 → 15 | +1 |
| intfloat/multilingual-e5-base | 67 → 66 | +1 | 57.02 → 58.28 | 43 → 42 | +1 |
| google/text-multilingual-embedding-002 | 33 → 33 | +0 | 62.16 → 63.74 | 29 → 28 | +1 |
| GritLM/GritLM-7B | 34 → 34 | +0 | 60.92 → 62.22 | 31 → 32 | -1 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 62 → 62 | +0 | 57.03 → 58.26 | 42 → 43 | -1 |
| codefuse-ai/F2LLM-v2-330M | 42 → 42 | +0 | 60.84 → 62.33 | 32 → 31 | +1 |

### Largest Borda rank movers (any coverage)

Large Borda moves are almost all models with partial task coverage: Borda only counts the tasks a model has results on, so losing a few removed tasks shifts them more.

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| mteb/baseline-bm25s | 215 → 241 | -26 | – → – | – → – | – |
| infly/inf-retriever-v1-1.5b | 221 → 240 | -19 | – → – | – → – | – |
| geevec-ai/geevec-embeddings-1.0 | 225 → 243 | -18 | – → – | – → – | – |
| mongodb/voyage-3-large | 218 → 235 | -17 | – → – | – → – | – |
| infly/inf-retriever-v1 | 214 → 231 | -17 | – → – | – → – | – |
| perplexity-ai/pplx-embed-v1-0.6b | 240 → 252 | -12 | – → – | – → – | – |
| perplexity-ai/pplx-embed-v1-4b | 235 → 245 | -10 | – → – | – → – | – |
| geevec-ai/geevec-embeddings-1.0-lite | 238 → 247 | -9 | – → – | – → – | – |
| dmis-lab/UltraBERT-Embed-150M | 205 → 213 | -8 | – → – | – → – | – |
| Querit/Querit-4B | 308 → 316 | -8 | – → – | – → – | – |
| Querit/Querit | 313 → 321 | -8 | – → – | – → – | – |
| KiteFishAI/Nano-Em1-0.6B-v2 | 241 → 234 | +7 | – → – | – → – | – |
| RiverRider/motherlode-code-small-en-v0.1 | 244 → 238 | +6 | – → – | – → – | – |
| prdev/mini-gte | 242 → 236 | +6 | – → – | – → – | – |
| LCO-Embedding/LCO-Embedding-Omni-7B | 245 → 239 | +6 | – → – | – → – | – |
| McGill-NLP/AfriE5-Large-instruct | 239 → 233 | +6 | – → – | – → – | – |
| ManiacLabs/miniac-embed | 243 → 237 | +6 | – → – | – → – | – |
| mongodb/voyage-code-2 | 230 → 225 | +5 | – → – | – → – | – |
| mixedbread-ai/deepset-mxbai-embed-de-large-v1 | 227 → 222 | +5 | – → – | – → – | – |
| facebook/SONAR | 228 → 223 | +5 | – → – | – → – | – |
## MTEB(Europe, v1) → MTEB(Europe, v2)

Models ranked: 387 (models with results on all tasks get a Mean(Task); Borda covers all models, as on the leaderboard).

- Borda rank: Kendall τ = 0.988, Spearman ρ = 0.999
- Mean(Task) rank (the 60 models complete on both): Kendall τ = 0.986
- Models that only become complete (get a Mean(Task)) in the new version, because they were missing just removed tasks: 0
- Models whose Borda rank changes: 226; median |Δ| = 1, max |Δ| = 41
- Top-10 by Borda (old): codefuse-ai/F2LLM-v2-14B, codefuse-ai/F2LLM-v2-8B, codefuse-ai/F2LLM-v2-4B, codefuse-ai/F2LLM-v2-1.7B, Linq-AI-Research/Linq-Embed-Mistral, Alibaba-NLP/gte-Qwen2-7B-instruct, Salesforce/SFR-Embedding-Mistral, GritLM/GritLM-7B, codefuse-ai/F2LLM-v2-0.6B, Qwen/Qwen3-Embedding-8B
- Top-10 by Borda (new): codefuse-ai/F2LLM-v2-14B, codefuse-ai/F2LLM-v2-8B, codefuse-ai/F2LLM-v2-4B, codefuse-ai/F2LLM-v2-1.7B, Linq-AI-Research/Linq-Embed-Mistral, Alibaba-NLP/gte-Qwen2-7B-instruct, Salesforce/SFR-Embedding-Mistral, GritLM/GritLM-7B, codefuse-ai/F2LLM-v2-0.6B, Qwen/Qwen3-Embedding-8B

### Top 50 (by either version), sorted by new Borda rank

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| codefuse-ai/F2LLM-v2-14B | 1 → 1 | +0 | 69.89 → 72.30 | 1 → 1 | +0 |
| codefuse-ai/F2LLM-v2-8B | 2 → 2 | +0 | 69.22 → 71.60 | 2 → 2 | +0 |
| codefuse-ai/F2LLM-v2-4B | 3 → 3 | +0 | 68.63 → 71.13 | 3 → 3 | +0 |
| codefuse-ai/F2LLM-v2-1.7B | 4 → 4 | +0 | 66.90 → 69.55 | 4 → 4 | +0 |
| Linq-AI-Research/Linq-Embed-Mistral | 5 → 5 | +0 | – → – | – → – | – |
| Alibaba-NLP/gte-Qwen2-7B-instruct | 6 → 6 | +0 | 63.60 → 65.83 | 6 → 6 | +0 |
| Salesforce/SFR-Embedding-Mistral | 7 → 7 | +0 | – → – | – → – | – |
| GritLM/GritLM-7B | 8 → 8 | +0 | 62.97 → 65.45 | 7 → 7 | +0 |
| codefuse-ai/F2LLM-v2-0.6B | 9 → 9 | +0 | 64.49 → 67.15 | 5 → 5 | +0 |
| Qwen/Qwen3-Embedding-8B | 10 → 10 | +0 | – → – | – → – | – |
| GritLM/GritLM-8x7B | 11 → 11 | +0 | 62.32 → 65.05 | 8 → 9 | -1 |
| intfloat/e5-mistral-7b-instruct | 13 → 12 | +1 | 61.73 → 64.40 | 12 → 12 | +0 |
| google/gemini-embedding-001 | 12 → 13 | -1 | – → – | – → – | – |
| intfloat/multilingual-e5-large-instruct | 17 → 14 | +3 | 62.19 → 64.66 | 10 → 11 | -1 |
| nvidia/llama-embed-nemotron-8b | 15 → 15 | +0 | – → – | – → – | – |
| Octen/Octen-Embedding-8B | 16 → 16 | +0 | – → – | – → – | – |
| microsoft/harrier-oss-v1-27b | 14 → 17 | -3 | – → – | – → – | – |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | 19 → 18 | +1 | – → – | – → – | – |
| Salesforce/SFR-Embedding-2_R | 21 → 19 | +2 | 62.29 → 65.13 | 9 → 8 | +1 |
| Qwen/Qwen3-Embedding-4B | 20 → 20 | +0 | – → – | – → – | – |
| Cohere/Cohere-embed-multilingual-v3.0 | 23 → 21 | +2 | – → – | – → – | – |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 18 → 22 | -4 | – → – | – → – | – |
| NovaSearch/jasper_en_vision_language_v1 | 22 → 23 | -1 | – → – | – → – | – |
| openai/text-embedding-3-large | 25 → 24 | +1 | – → – | – → – | – |
| jinaai/jina-embeddings-v5-omni-small | 27 → 25 | +2 | – → – | – → – | – |
| NovaSearch/stella_en_1.5B_v5 | 29 → 26 | +3 | 59.81 → 62.39 | 14 → 14 | +0 |
| Lajavaness/bilingual-embedding-large | 32 → 27 | +5 | – → – | – → – | – |
| jinaai/jina-embeddings-v5-text-small | 28 → 28 | +0 | – → – | – → – | – |
| jinaai/jina-embeddings-v3 | 36 → 29 | +7 | 60.72 → 63.81 | 13 → 13 | +0 |
| nvidia/NV-Embed-v2 | 26 → 30 | -4 | – → – | – → – | – |
| codefuse-ai/F2LLM-v2-330M | 31 → 31 | +0 | 62.04 → 64.76 | 11 → 10 | +1 |
| microsoft/harrier-oss-v1-0.6b | 24 → 32 | -8 | – → – | – → – | – |
| jinaai/jina-embeddings-v5-omni-nano | 34 → 33 | +1 | – → – | – → – | – |
| jinaai/jina-embeddings-v5-text-nano | 35 → 34 | +1 | – → – | – → – | – |
| Bytedance/Seed1.6-embedding-1215 | 30 → 35 | -5 | – → – | – → – | – |
| ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826 | 33 → 36 | -3 | – → – | – → – | – |
| ai-sage/Giga-Embeddings-instruct-3B-0826 | 37 → 37 | +0 | – → – | – → – | – |
| ICT-TIME-and-Querit/ICT-TIME-and-Querit-embedding-v1 | 39 → 38 | +1 | – → – | – → – | – |
| BidirLM/BidirLM-Omni-2.5B-Embedding | 38 → 39 | -1 | – → – | – → – | – |
| nvidia/NV-Embed-v1 | 40 → 40 | +0 | – → – | – → – | – |
| OrdalieTech/Solon-embeddings-large-0.1 | 43 → 41 | +2 | 59.40 → 61.96 | 15 → 15 | +0 |
| Qwen/Qwen3-Embedding-0.6B | 41 → 42 | -1 | – → – | – → – | – |
| intfloat/multilingual-e5-large | 45 → 43 | +2 | – → – | – → – | – |
| ICT-TIME-and-Querit/BOOM_4B_v1 | 46 → 44 | +2 | – → – | – → – | – |
| HIT-TMG/KaLM-embedding-multilingual-mini-v1 | 44 → 45 | -1 | – → – | – → – | – |
| Alibaba-NLP/gte-multilingual-base | 50 → 46 | +4 | – → – | – → – | – |
| HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1 | 48 → 47 | +1 | – → – | – → – | – |
| BidirLM/BidirLM-1.7B-Embedding | 49 → 48 | +1 | – → – | – → – | – |
| manu/bge-m3-custom-fr | 47 → 49 | -2 | 58.92 → 61.19 | 18 → 18 | +0 |
| microsoft/harrier-oss-v1-270m | 42 → 50 | -8 | – → – | – → – | – |

### Largest Mean(Task) rank movers (models complete on both versions)

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet | 112 → 111 | +1 | 51.72 → 53.93 | 29 → 27 | +2 |
| nomic-ai/nomic-embed-text-v1.5 | 120 → 121 | -1 | 46.32 → 48.06 | 45 → 47 | -2 |
| manu/sentence_croissant_alpha_v0.2 | 125 → 125 | +0 | 49.23 → 51.38 | 38 → 36 | +2 |
| hotchpotch/bekko-embedding-v1-a25m | 53 → 54 | -1 | 59.02 → 61.49 | 17 → 16 | +1 |
| codefuse-ai/F2LLM-v2-160M | 52 → 53 | -1 | 59.06 → 61.46 | 16 → 17 | -1 |
| sentence-transformers/LaBSE | 95 → 95 | +0 | 51.84 → 53.88 | 27 → 28 | -1 |
| intfloat/e5-base-v2 | 89 → 94 | -5 | 49.41 → 51.36 | 36 → 37 | -1 |
| sdadas/mmlw-roberta-base | 114 → 113 | +1 | 46.16 → 48.48 | 46 → 45 | +1 |
| BAAI/bge-small-en-v1.5 | 117 → 114 | +3 | 46.09 → 48.41 | 47 → 46 | +1 |
| intfloat/e5-small-v2 | 115 → 117 | -2 | 47.44 → 49.16 | 42 → 43 | -1 |
| manu/sentence_croissant_alpha_v0.4 | 121 → 120 | +1 | 49.71 → 51.68 | 33 → 32 | +1 |
| WhereIsAI/UAE-Large-V1 | 72 → 75 | -3 | 49.54 → 51.46 | 34 → 35 | -1 |
| mixedbread-ai/mxbai-embed-large-v1 | 73 → 74 | -1 | 49.30 → 51.30 | 37 → 38 | -1 |
| manu/sentence_croissant_alpha_v0.3 | 122 → 122 | +0 | 49.49 → 51.58 | 35 → 34 | +1 |
| sentence-transformers/static-similarity-mrl-multilingual-v1 | 126 → 130 | -4 | 49.77 → 51.62 | 32 → 33 | -1 |

### Largest Borda rank movers (any coverage)

Large Borda moves are almost all models with partial task coverage: Borda only counts the tasks a model has results on, so losing a few removed tasks shifts them more.

| Model | Borda old → new | Δ | Mean(Task) old → new | Mean(Task) rank old → new | Δ |
|---|---:|---:|---:|---:|---:|
| geevec-ai/geevec-embeddings-1.0-lite | 215 → 256 | -41 | – → – | – → – | – |
| perplexity-ai/pplx-embed-v1-0.6b | 221 → 261 | -40 | – → – | – → – | – |
| perplexity-ai/pplx-embed-v1-4b | 216 → 251 | -35 | – → – | – → – | – |
| geevec-ai/geevec-embeddings-1.0 | 206 → 238 | -32 | – → – | – → – | – |
| Querit/Querit | 319 → 335 | -16 | – → – | – → – | – |
| mteb/baseline-bm25s | 199 → 213 | -14 | – → – | – → – | – |
| Querit/Querit-4B | 316 → 330 | -14 | – → – | – → – | – |
| jinaai/jina-reranker-v3 | 321 → 331 | -10 | – → – | – → – | – |
| dmis-lab/UltraBERT-Embed-150M | 202 → 212 | -10 | – → – | – → – | – |
| microsoft/harrier-oss-v1-0.6b | 24 → 32 | -8 | – → – | – → – | – |
| microsoft/harrier-oss-v1-270m | 42 → 50 | -8 | – → – | – → – | – |
| zeroentropy/zembed-1 | 101 → 109 | -8 | – → – | – → – | – |
| omarelshehy/arabic-english-sts-matryoshka | 77 → 70 | +7 | – → – | – → – | – |
| jinaai/jina-embeddings-v3 | 36 → 29 | +7 | 60.72 → 63.81 | 13 → 13 | +0 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | 134 → 140 | -6 | – → – | – → – | – |
| ibm-granite/granite-embedding-311m-multilingual-r2 | 113 → 119 | -6 | – → – | – → – | – |
| nomic-ai/nomic-embed-text-v1 | 91 → 97 | -6 | – → – | – → – | – |
| google/text-embedding-004 | 203 → 198 | +5 | – → – | – → – | – |
| Lajavaness/bilingual-embedding-large | 32 → 27 | +5 | – → – | – → – | – |
| mongodb/voyage-3-large | 197 → 202 | -5 | – → – | – → – | – |

# Pareto frontier changes (Mean(Task) vs. model size)

A model is on the Pareto frontier if no other model has at most as many parameters and at least as high a Mean(Task) (strictly better in one).
Only models with results on all tasks (a Mean(Task)) and a known parameter count are included, so API models without a published size are excluded.

## MTEB(Multilingual, v2) → MTEB(Multilingual, v3)

### Size = Total Parameters (B)

Models considered: 83 (old), 83 (new). Frontier size: 15 → 14. Kept: 14, dropped: 1, added: 0.

| Model | Params (B) | Mean(Task) old | Mean(Task) new | Status |
|---|---:|---:|---:|---|
| mteb/baseline-random-encoder | 0.000 | 18.79 | 19.12 | kept |
| Mihaiii/gte-micro | 0.017 | 39.42 | 40.04 | kept |
| Mihaiii/gte-micro-v4 | 0.019 | 41.53 | 42.33 | kept |
| intfloat/e5-small-v2 | 0.033 | 44.47 | 45.25 | kept |
| codefuse-ai/F2LLM-v2-80M | 0.080 | 55.23 | 56.51 | kept |
| hotchpotch/bekko-embedding-v1-a8m | 0.106 | 56.73 | 57.88 | kept |
| hotchpotch/bekko-embedding-v1-a25m | 0.123 | 58.36 | 59.71 | kept |
| jinaai/jina-embeddings-v5-text-nano | 0.212 | 65.52 | 67.13 | kept |
| microsoft/harrier-oss-v1-270m | 0.268 | 66.55 | 67.40 | kept |
| microsoft/harrier-oss-v1-0.6b | 0.596 | 69.01 | 69.52 | kept |
| Qwen/Qwen3-Embedding-4B | 4.022 | 69.45 | 71.05 | kept |
| nvidia/llama-embed-nemotron-8b | 7.505 | 69.46 | 71.00 | **dropped** (now dominated by Qwen/Qwen3-Embedding-4B: 4.022B, 71.05) |
| Qwen/Qwen3-Embedding-8B | 7.567 | 70.58 | 72.23 | kept |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 11.766 | 72.32 | 73.10 | kept |
| microsoft/harrier-oss-v1-27b | 27.009 | 74.27 | 74.74 | kept |

### Size = Active Parameters (B)

Models considered: 80 (old), 80 (new). Frontier size: 9 → 9. Kept: 9, dropped: 0, added: 0.

| Model | Params (B) | Mean(Task) old | Mean(Task) new | Status |
|---|---:|---:|---:|---|
| sentence-transformers/static-similarity-mrl-multilingual-v1 | 0.000 | 47.21 | 48.09 | kept |
| hotchpotch/bekko-embedding-v1-a8m | 0.008 | 56.73 | 57.88 | kept |
| hotchpotch/bekko-embedding-v1-a25m | 0.025 | 58.36 | 59.71 | kept |
| microsoft/harrier-oss-v1-270m | 0.100 | 66.55 | 67.40 | kept |
| microsoft/harrier-oss-v1-0.6b | 0.440 | 69.01 | 69.52 | kept |
| Qwen/Qwen3-Embedding-4B | 3.634 | 69.45 | 71.05 | kept |
| Qwen/Qwen3-Embedding-8B | 6.946 | 70.58 | 72.23 | kept |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 10.759 | 72.32 | 73.10 | kept |
| microsoft/harrier-oss-v1-27b | 25.599 | 74.27 | 74.74 | kept |

## MTEB(Europe, v1) → MTEB(Europe, v2)

### Size = Total Parameters (B)

Models considered: 60 (old), 60 (new). Frontier size: 16 → 14. Kept: 14, dropped: 2, added: 0.

| Model | Params (B) | Mean(Task) old | Mean(Task) new | Status |
|---|---:|---:|---:|---|
| mteb/baseline-random-encoder | 0.000 | 15.20 | 16.05 | kept |
| Mihaiii/gte-micro | 0.017 | 40.79 | 42.20 | kept |
| Mihaiii/gte-micro-v4 | 0.019 | 42.93 | 44.65 | kept |
| sentence-transformers/all-MiniLM-L6-v2 | 0.023 | 43.45 | 45.12 | kept |
| intfloat/e5-small-v2 | 0.033 | 47.44 | 49.16 | kept |
| codefuse-ai/F2LLM-v2-80M | 0.080 | 56.24 | 58.60 | kept |
| hotchpotch/bekko-embedding-v1-a8m | 0.106 | 57.52 | 59.62 | kept |
| hotchpotch/bekko-embedding-v1-a25m | 0.123 | 59.02 | 61.49 | kept |
| codefuse-ai/F2LLM-v2-160M | 0.159 | 59.06 | 61.46 | **dropped** (now dominated by hotchpotch/bekko-embedding-v1-a25m: 0.123B, 61.49) |
| codefuse-ai/F2LLM-v2-330M | 0.334 | 62.04 | 64.76 | kept |
| intfloat/multilingual-e5-large-instruct | 0.560 | 62.19 | 64.66 | **dropped** (now dominated by codefuse-ai/F2LLM-v2-330M: 0.334B, 64.76) |
| codefuse-ai/F2LLM-v2-0.6B | 0.596 | 64.49 | 67.15 | kept |
| codefuse-ai/F2LLM-v2-1.7B | 1.721 | 66.90 | 69.55 | kept |
| codefuse-ai/F2LLM-v2-4B | 4.022 | 68.63 | 71.13 | kept |
| codefuse-ai/F2LLM-v2-8B | 7.568 | 69.22 | 71.60 | kept |
| codefuse-ai/F2LLM-v2-14B | 13.990 | 69.89 | 72.30 | kept |

### Size = Active Parameters (B)

Models considered: 57 (old), 57 (new). Frontier size: 11 → 9. Kept: 9, dropped: 2, added: 0.

| Model | Params (B) | Mean(Task) old | Mean(Task) new | Status |
|---|---:|---:|---:|---|
| sentence-transformers/static-similarity-mrl-multilingual-v1 | 0.000 | 49.77 | 51.62 | kept |
| hotchpotch/bekko-embedding-v1-a8m | 0.008 | 57.52 | 59.62 | kept |
| hotchpotch/bekko-embedding-v1-a25m | 0.025 | 59.02 | 61.49 | kept |
| codefuse-ai/F2LLM-v2-160M | 0.062 | 59.06 | 61.46 | **dropped** (now dominated by hotchpotch/bekko-embedding-v1-a25m: 0.025B, 61.49) |
| codefuse-ai/F2LLM-v2-330M | 0.198 | 62.04 | 64.76 | kept |
| intfloat/multilingual-e5-large-instruct | 0.304 | 62.19 | 64.66 | **dropped** (now dominated by codefuse-ai/F2LLM-v2-330M: 0.198B, 64.76) |
| codefuse-ai/F2LLM-v2-0.6B | 0.441 | 64.49 | 67.15 | kept |
| codefuse-ai/F2LLM-v2-1.7B | 1.410 | 66.90 | 69.55 | kept |
| codefuse-ai/F2LLM-v2-4B | 3.634 | 68.63 | 71.13 | kept |
| codefuse-ai/F2LLM-v2-8B | 6.947 | 69.22 | 71.60 | kept |
| codefuse-ai/F2LLM-v2-14B | 13.214 | 69.89 | 72.30 | kept |
