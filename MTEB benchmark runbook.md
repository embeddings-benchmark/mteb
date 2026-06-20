# MTEB benchmark runbook

> **Update (2026-06-19) — read first; where this conflicts with sections below, this wins.**
> The doc below was an earlier snapshot (BidirLM-Omni-centric, consolidated from the former
> `MTEB.md`, `MTEB-BidirLM-Omni.md`, `MTEB-model-coverage.md`, `MTEB-dataset-sizes.md` — those
> internal "see X.md" links now refer to sections *within this file*). Corrections since then:
>
> - **Repo / sync:** canonical remote is now `JSALT2026-OmniEnc/mteb` (`origin`); fully synced
>   with `embeddings-benchmark/mteb` upstream (`e091c0d4`, 2026-06-19), **10 commits ahead** with
>   the WAVE-7B integration on branch `wave-7b-integration`.
> - **Python:** the login shell now reports **3.13.4** and the project venv uses **3.10** — both
>   satisfy `pyproject.toml`'s `>=3.10,<3.15`. (The "Python 3.9.21" notes below are obsolete.)
> - **Video/omni benchmarks now EXIST** (the "no named MVEB benchmark / video gap" claims below
>   are obsolete): **`MVEB(beta)` = 23 tasks**, `MVEB(video, beta)` = 9, `MVEB(text, video, beta)`
>   = 19, all in `mteb/benchmarks/benchmarks/benchmarks.py`.
> - **Audio benchmarks currently registered:** `MAEB(beta)` = 30 tasks, `MAEB(beta, audio-only)`
>   = 19. `MAEB(beta, extended)` and `MAEB+(beta)` referenced below are **not** in the current
>   registry.
> - **Model registry:** ~**733 `ModelMeta` across 196 implementation files** (was 716/192).
> - **WAVE-7B** (`tsinghua-ee/WAVE-7B`, omni text/audio/video, 3584-d) is now a registered model.
>   See **`MTEB-WAVE-7B.md`** for the integration + the numerical faithfulness parity harness
>   (`scripts/validate_wave_faithfulness.py`). WAVE benchmark runs on `MVEB(beta)`/`MAEB(beta)`
>   are in progress.
> - **Dataset-size tables below are a dated HF snapshot** — regenerate before relying on exact
>   bytes; rows for the removed `MAEB(beta, extended)`/`MAEB+(beta)` benchmarks no longer apply.

**MTEB benchmark runbook**  
**MTEB** is an evaluation framework and registry for embedding models. The same package now covers text MTEB, image/image-text MIEB, audio/audio-text MAEB, and individual video/audio-video tasks. Our team will use MTEB by selecting task or benchmark objects, providing a compatible model object, and collecting per-task JSON results.

Related writeups: BidirLM worked example (MTEB-BidirLM-Omni.md), model coverage snapshot (MTEB-model-coverage.md), and dataset size inventory (MTEB-dataset-sizes.md).

**At a glance**

| Team question | Answer |
| :---- | :---- |
| Does it download evaluation data automatically? | **Yes** for built-in tasks. AbsTask.load\_data() calls datasets.load\_dataset(TaskMetadata.dataset). Retrieval tasks call RetrievalDatasetLoader, which loads corpus, queries, qrels/default, optional instruction, and optional top\_ranked configs from Hugging Face. |
| Do private evals need curated data? | **Yes**. A private eval still needs a task-loadable dataset. Use a Hugging Face Hub dataset repo, load\_dataset("json" / "csv" / "parquet", data\_files=...) for local or remote files, or a custom load\_data() that builds Dataset.from\_dict(...) / RetrievalSplitData. |
| What is the data size on disk? | There is **no** **one** MTEB size, but we can query exact remote repo file sizes per task. The generated inventory covers 1,759 non-aggregate task rows and 1,468 unique HF dataset repo/revision refs. Local cache size must still be measured after a run because HF repo bytes and materialized Arrow/cache bytes can differ. |
| How do we provide a new model? | Pass mteb.get\_model(...), a SentenceTransformer / CrossEncoder, a search/index wrapper, or a custom object implementing MTEB's encode(...), predict(...), or index/search(...) protocols. For first-class support, add a ModelMeta under mteb/mteb/models/model\_implementations/. |
| What model types are compatible? | Dense encoders, cross-encoders, search/index wrappers, late-interaction models, sparse/router models, API-backed wrappers, and multimodal encoders over text, image, audio, and/or video batch fields. |
| How do we add tasks/evals? | Subclass an AbsTask, fill TaskMetadata, provide an HF-backed dataset or custom load\_data(), compute descriptive stats, add tests, and optionally group tasks into a Benchmark. |
| What compute is required? | MTEB does not define a universal minimum GPU. Compute is model plus task plus modality dependent. Use ModelMeta.memory\_usage\_mb for model load estimates, then run the pilot commands below to record GPU, batch size, wall time, evaluation\_time, and cache growth. |

**Run command**  
Use Python \>=3.10 for the local clone. This shell currently reports Python 3.9.21, while mteb/pyproject.toml declares Python \>=3.10,\<3.15.

\`\`\`

pip install mteb

pip install "mteb\[image,audio,video\]"

\`\`\`

MTEB's install docs call out extra audio requirements when using datasets\>=4: FFmpeg and transformers\>=4.57.6. The video extra pulls video decoding support.

\`\`\`

import mteb

model \= mteb.get\_model("sentence-transformers/all-MiniLM-L6-v2")

tasks \= mteb.get\_benchmark("MTEB(eng, v2)")

results \= mteb.evaluate(model, tasks=tasks, encode\_kwargs={"batch\_size": 32})

mteb run \\

  \-m sentence-transformers/all-MiniLM-L6-v2 \\

  \-t ArguAna \\

  \--batch-size 32 \\

  \--output-folder results

\`\`\`

Useful benchmark names source-checked against mteb/mteb/benchmarks/benchmarks/benchmarks.py:

| Family | Benchmark names |
| :---- | :---- |
| Text | MTEB(eng, v2), MTEB(Multilingual, v2) |
| Image/image-text | MIEB(lite), MIEB(eng), MIEB(Multilingual), MIEB(Img) |
| Audio/audio-text | MAEB(beta), MAEB(beta, audio-only), MAEB(beta, extended), MAEB+(beta) |
| Video/omni | Video and mixed-modality tasks exist locally, but MVEB(beta) (23 tasks), MVEB(video, beta) (9), MVEB(text, video, beta) (19). [Updated 2026-06-19; the "no MVEB benchmark" claim is obsolete.] |

**Dataset loading**  
Built-in non-retrieval tasks use the default AbsTask.load\_data() path:

self.dataset \= load\_dataset(\*\*self.metadata.dataset, num\_proc=num\_proc)

For multilingual tasks, MTEB iterates over self.hf\_subsets and calls:

load\_dataset(name=hf\_subset, \*\*self.metadata.dataset, num\_proc=num\_proc)

That means TaskMetadata.dataset can carry normal Hugging Face datasets.load\_dataset arguments such as path, revision, split, and task-specific config/subset values.

Retrieval tasks use AbsTaskRetrieval.load\_data() and RetrievalDatasetLoader. A retrieval dataset repo/config must expose:

| Piece | Required columns / shape |
| :---- | :---- |
| corpus config | id plus at least one modality column such as text, image, audio, or video; text corpora may include title. |
| queries config | id plus at least one modality column; instruction retrieval can include instruction. |
| default or qrels config | query-id, corpus-id, score. |
| optional instruction config | Joined into queries when present. |
| optional top\_ranked config | query-id, corpus-ids; used for reranking. |

Private/local eval options:

from datasets import Dataset, load\_dataset

\# HF Hub dataset repo, pinned by revision.

ds \= load\_dataset("my-org/my-private-eval", revision="0123abcd")

\# Local files through the normal HF datasets loaders.

ds \= load\_dataset("json", data\_files={"test": "/secure/evals/test.jsonl"})

ds \= load\_dataset("parquet", data\_files={"test": "/secure/evals/test.parquet"})

\# In-memory construction for custom load\_data().

queries \= Dataset.from\_dict({"id": \["q1"\], "text": \["find the matching item"\]})

Use a Hub dataset repo when the task should be shareable or repeatable in CI. Use local files or in-memory construction when the data cannot leave the environment.

**Dataset size inventory**  
The sidecar MTEB-dataset-sizes.md was generated from local task metadata plus live Hugging Face Hub metadata:

* Source task query target: mteb.get\_tasks(exclude\_superseded=False, exclude\_private=False, exclude\_beta=False, exclude\_aggregate=True); because this shell Python is 3.9, the sidecar uses an equivalent static parse of non-aggregate TaskMetadata records.  
* Size source: HfApi().dataset\_info(path, revision=revision, files\_metadata=True).  
* HF repo bytes means exact summed Hub file sizes visible to this environment for that dataset repo/revision.  
* local cache bytes means exact disk use only after loading/running, measured with du.

Interpret task names carefully. TaskMetadata.modalities is the source of truth for modality, not the task name. For example, VideoRetrieval sounds like a video benchmark, but the local task declares category="t2t" and modalities=\["text"\]; its 6.99 MiB HF repo contains text/parquet files for retrieving video titles. Actual video payload retrieval tasks are rows with video in the Modalities column, usually Any2AnyRetrieval. Also, HF repo bytes only measures files stored in the HF dataset repo; if a dataset stores metadata that references external media, final local disk use can be larger after loading or preprocessing.

Aggregate remote size snapshot:

| Modality cluster | Task rows | Unique dataset refs | Sized refs | HF repo bytes | HF repo size |
| :---- | :---- | :---- | :---- | :---- | :---- |
| text | 1,247 | 1,135 | 1,117 | 379,347,007,436 | 353.29 GiB |
| image/image-text | 217 | 187 | 183 | 876,292,711,490 | 816.11 GiB |
| audio/audio-text | 110 | 95 | 95 | 3,519,401,949,423 | 3.20 TiB |
| video/video-text/audio-video | 185 | 51 | 51 | 474,770,253,696 | 442.16 GiB |

Benchmark remote size snapshot:

| Benchmark | Benchmark tasks | Matched non-aggregate rows | Unique dataset refs | Sized refs | HF repo size |
| :---- | :---- | :---- | :---- | :---- | :---- |
| MTEB(eng, v2) | 41 | 41 | 41 | 41 | 3.71 GiB |
| MTEB(Multilingual, v2) | 131 | 131 | 131 | 131 | 8.90 GiB |
| MIEB(lite) | 51 | 49 | 44 | 44 | 71.92 GiB |
| MIEB(eng) | 125 | 123 | 96 | 96 | 529.79 GiB |
| MIEB(Multilingual) | 130 | 126 | 99 | 99 | 536.96 GiB |
| MIEB(Img) | 49 | 49 | 47 | 47 | 383.31 GiB |
| MAEB(beta) | 30 | 30 | 28 | 28 | 378.85 GiB |
| MAEB(beta, audio-only) | 19 | 19 | 18 | 18 | 22.27 GiB |
| MAEB(beta, extended) | 89 | 89 | 77 | 77 | 739.08 GiB |
| MAEB+(beta) | 98 | 98 | 83 | 83 | 3.15 TiB |

Examples from the inventory:

| Task | Dataset repo | HF repo size |
| :---- | :---- | :---- |
| PoemSentimentClassification.v2 | mteb/poem\_sentiment | 50,188 bytes / 49.01 KiB |
| CQADupstackAndroidRetrieval | mteb/CQADupstackAndroidRetrieval | 8,498,683 bytes / 8.10 MiB |
| ArguAna | mteb/arguana | 11,730,754 bytes / 11.19 MiB |
| Country211 | mteb/wds\_country211 | 9,554,508,253 bytes / 8.90 GiB |
| CommonLanguageAgeDetection | mteb/commonlanguage-age-mini | 776,262,254 bytes / 740.30 MiB |

Misleading-name examples:

| Task | Modalities | Interpretation | HF repo size |
| :---- | :---- | :---- | :---- |
| VideoRetrieval | text | Text retrieval in a video-title domain, not video-media retrieval. | 6.99 MiB |
| ActivityNetCaptionsT2VRetrieval | video,text | Actual text-to-video retrieval with video payloads in the repo. | 12.83 GiB |
| VATEXT2VRetrieval | text,video | Actual text-to-video retrieval with video payloads in the repo. | 21.77 GiB |

Measure exact local disk after a pilot:

export HF\_HOME="$PWD/.hf-mteb-pilot"

du \-sh "$HF\_HOME" \~/.cache/mteb results 2\>/dev/null

**Model integration**  
Existing registry model:

import mteb

model \= mteb.get\_model("BidirLM/BidirLM-Omni-2.5B-Embedding")

results \= mteb.evaluate(

    model,

    tasks=mteb.get\_tasks(tasks=\["ArguAna"\]),

    encode\_kwargs={"batch\_size": 4},

)

Local or Hub SentenceTransformers model:

import mteb

from sentence\_transformers import SentenceTransformer

model \= SentenceTransformer("/models/my-encoder")  \# or a HF model id

tasks \= mteb.get\_tasks(tasks=\["ArguAna"\])

results \= mteb.evaluate(model, tasks=tasks, encode\_kwargs={"batch\_size": 32})

Custom encoder wrapper:

from torch.utils.data import DataLoader

from mteb.types import BatchedInput

class MyEncoder:

    def encode(

        self,

        inputs: DataLoader\[BatchedInput\],

        \*,

        task\_metadata,

        hf\_split: str,

        hf\_subset: str,

        prompt\_type=None,

        \*\*kwargs,

    ):

        batch\_size \= kwargs.get("batch\_size", 32\)

        \# Iterate over MTEB batches containing text/image/audio/video fields.

        \# Return numpy or torch embeddings shaped \[num\_inputs, embedding\_dim\].

        ...

First-class registry path:

* Add a file or entry under mteb/mteb/models/model\_implementations/.  
* Define a ModelMeta with name, loader, revision, embed\_dim, memory\_usage\_mb, modalities, model\_type, dependencies, and citations.  
* BidirLM's local metadata is the concrete reference: BidirLM/BidirLM-Omni-2.5B-Embedding, 2,445,009,536 parameters, 2048-dimensional embeddings, memory\_usage\_mb=4663, modalities text,image,audio,video, and model\_type=\["dense"\].

**Compatible model types**

| Type | MTEB interface | Where it fits |
| :---- | :---- | :---- |
| Dense encoders | EncoderProtocol.encode(...) | Standard embedding tasks over text and multimodal inputs. |
| Cross-encoders | CrossEncoderProtocol.predict(...) | Reranking and pairwise scoring tasks. |
| Search/index wrappers | SearchProtocol.index(...) and search(...) | Retrieval systems that manage their own index or vector database. |
| Late-interaction models | Registered ModelMeta.model\_type=\["late-interaction"\] plus custom loader | ColBERT/ColPali-style token-level retrieval. |
| Sparse/router models | model\_type=\["sparse"\] or \["router"\] | Sparse retrieval and routed embedding systems. |
| API-backed wrappers | Custom ModelMeta.loader or wrapper class | Cohere, Voyage, Gemini, OpenAI-style services where local checkpoint files are not required. |
| Multimodal encoders | encode(...) over BatchedInput fields | Image, audio, video, and omni encoders. |

Model discovery by modality:

import mteb

metas \= mteb.get\_model\_metas(

    modalities=\["image", "text"\],

    exclusive\_modality\_filter=True,

)

names \= \[(m.name, m.modalities, m.model\_type) for m in metas\]

Current local model coverage: MTEB-model-coverage.md. A source parse of local ModelMeta(...) records found 716 registrations. Known memory\_usage\_mb ranges by model modality cluster:

| Model modality cluster | ModelMeta rows | Known memory rows | p50 memory | p90 memory | Max memory |
| :---- | :---- | :---- | :---- | :---- | :---- |
| text | 508 | 427 | 830 MB | 13.5 GiB | 87.0 GiB |
| image/image-text | 127 | 116 | 4.5 GiB | 17.5 GiB | 32.9 GiB |
| audio/audio-text | 53 | 52 | 675 MB | 3.9 GiB | 9.0 GiB |
| video/video-text/audio-video | 8 | 8 | 3.9 GiB | 4.2 GiB | 4.2 GiB |
| omni/mixed | 20 | 19 | 8.8 GiB | 59.1 GiB | 65.7 GiB |

Treat these as model load metadata, not guaranteed end-to-end peak memory. Retrieval can require much more runtime storage because the corpus embeddings and index dominate.

**Adding tasks/evals**  
HF-backed private retrieval task:

import mteb

from mteb.abstasks import AbsTaskRetrieval

class MyPrivateRetrievalTask(AbsTaskRetrieval):

    metadata \= mteb.TaskMetadata(

        name="MyPrivateRetrievalTask",

        description="Private text-image retrieval smoke test.",

        reference=None,

        dataset={"path": "my-org/private-retrieval", "revision": "0123abcd"},

        type="Retrieval",

        category="t2i",

        modalities=\["text", "image"\],

        eval\_splits=\["test"\],

        eval\_langs=\["eng-Latn"\],

        main\_score="ndcg\_at\_10",

        date=("2026-01-01", "2026-01-01"),

        domains=\["Web"\],

        task\_subtypes=\[\],

        license="not specified",

        annotations\_creators="derived",

        dialect=\[\],

        sample\_creation="created",

        bibtex\_citation="",

        is\_public=False,

    )

model \= mteb.get\_model("BidirLM/BidirLM-Omni-2.5B-Embedding")

results \= mteb.evaluate(

    model,

    tasks=\[MyPrivateRetrievalTask()\],

    encode\_kwargs={"batch\_size": 4},

)

Local in-memory retrieval task when data cannot be uploaded:

import mteb

from datasets import Dataset

from mteb.abstasks import AbsTaskRetrieval

from mteb.abstasks.retrieval\_dataset\_loaders import RetrievalSplitData

class MyLocalRetrievalTask(AbsTaskRetrieval):

    metadata \= mteb.TaskMetadata(

        name="MyLocalRetrievalTask",

        description="Private local retrieval task.",

        reference=None,

        dataset={"path": "local/in-memory", "revision": "local"},

        type="Retrieval",

        category="t2t",

        modalities=\["text"\],

        eval\_splits=\["test"\],

        eval\_langs=\["eng-Latn"\],

        main\_score="ndcg\_at\_10",

        date=("2026-01-01", "2026-01-01"),

        domains=\["Web"\],

        task\_subtypes=\["Article retrieval"\],

        license="not specified",

        annotations\_creators="derived",

        dialect=\[\],

        sample\_creation="created",

        bibtex\_citation="",

        is\_public=False,

    )

    def load\_data(self, num\_proc=None, \*\*kwargs):

        if self.data\_loaded:

            return

        self.dataset \= {

            "default": {

                "test": RetrievalSplitData(

                    corpus=Dataset.from\_dict({

                        "id": \["d1", "d2"\],

                        "text": \["first document", "second document"\],

                    }),

                    queries=Dataset.from\_dict({

                        "id": \["q1"\],

                        "text": \["find the first document"\],

                    }),

                    relevant\_docs={"q1": {"d1": 1}},

                    top\_ranked=None,

                )

            }

        }

        self.data\_loaded \= True

Contribution checklist for upstream-quality tasks:

1. Subclass the matching task base, for example AbsTaskRetrieval, AbsTaskClassification, or AbsTaskSTS.  
2. Fill TaskMetadata, including dataset, eval\_splits, eval\_langs, modalities, main\_score, license, domains, and public/private/beta flags.  
3. Use a pinned HF dataset revision when possible.  
4. Run task.load\_data() locally and inspect one example.  
5. Compute descriptive statistics with the MTEB helper before contributing.  
6. Add tests and add the task to a Benchmark only when it should be part of a reusable suite.

**Compute and pilot plan**  
MTEB has no universal minimum GPU type. Concrete planning needs the exact model, task set, batch size, and modality.

Grounded examples:

* BidirLM local metadata says 2.445B parameters, 2048-dimensional embeddings, memory\_usage\_mb=4663, and full text,image,audio,video modality support.  
* BidirLM HF repository size is separately documented in the worked example as about 4.91 GB, with model.safetensors about 4.89 GB.  
* Public MTEB result JSONs include evaluation\_time, but hardware is not consistently recorded. Examples: sentence-transformers/all-MiniLM-L6-v2 on ArguAna reports 6.81s; the same model on MSMARCO reports 79,308s, about 22.0h; intfloat/multilingual-e5-small on NanoNQRetrieval reports 16.68s.

Those runtime examples were read from public result JSONs: ArguAna (https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/sentence-transformers\_\_all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a/ArguAna.json), MSMARCO (https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/sentence-transformers\_\_all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a/MSMARCO.json), and NanoNQRetrieval (https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/intfloat\_\_multilingual-e5-small/fd1525a9fd15316a2d503bf26ab031a61d056e98/NanoNQRetrieval.json).

Run these pilots before scheduling full benchmarks:

export HF\_HOME="$PWD/.hf-mteb-pilot"

mkdir \-p results/mteb-pilot

nvidia-smi \--query-gpu=name,memory.total \--format=csv

/usr/bin/time \-v mteb run \\

  \-m BidirLM/BidirLM-Omni-2.5B-Embedding \\

  \-t ArguAna \\

  \--batch-size 4 \\

  \--output-folder results/mteb-pilot \\

  \--verbosity 2

du \-sh "$HF\_HOME" results/mteb-pilot \~/.cache/mteb 2\>/dev/null

/usr/bin/time \-v mteb run \\

  \-m BidirLM/BidirLM-Omni-2.5B-Embedding \\

  \-t Country211 \\

  \--batch-size 4 \\

  \--output-folder results/mteb-pilot \\

  \--verbosity 2

/usr/bin/time \-v mteb run \\

  \-m BidirLM/BidirLM-Omni-2.5B-Embedding \\

  \-t CommonLanguageAgeDetection \\

  \--batch-size 4 \\

  \--output-folder results/mteb-pilot \\

  \--verbosity 2

For each pilot, record:

* GPU name and memory from nvidia-smi.  
* Batch size and installed package versions.  
* Wall time from /usr/bin/time \-v.  
* Per-task evaluation\_time from the result JSON.  
* du \-sh "$HF\_HOME" and result/cache folder size after the run.

**Sources**

* MTEB docs: installation (https://embeddings-benchmark.github.io/mteb/installation/), running evaluations (https://embeddings-benchmark.github.io/mteb/get\_started/usage/running\_the\_evaluation/), defining models (https://embeddings-benchmark.github.io/mteb/get\_started/usage/defining\_the\_model/), adding tasks (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_dataset/), adding models (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_model/), adding benchmarks (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_benchmark/), and available benchmarks (https://embeddings-benchmark.github.io/mteb/overview/available\_benchmarks/).  
* Hugging Face docs: loading datasets (https://huggingface.co/docs/datasets/loading), cache management (https://huggingface.co/docs/datasets/cache), and \`HfApi.dataset\_info\` (https://huggingface.co/docs/huggingface\_hub/package\_reference/hf\_api\#huggingface\_hub.HfApi.dataset\_info).  
* Local source checked: mteb/mteb/abstasks/abstask.py, mteb/mteb/abstasks/retrieval.py, mteb/mteb/abstasks/retrieval\_dataset\_loaders.py, mteb/mteb/models/models\_protocols.py, mteb/mteb/models/get\_model\_meta.py, mteb/mteb/models/model\_meta.py, mteb/mteb/models/model\_implementations/bidirlm\_omni\_models.py, and mteb/mteb/benchmarks/benchmarks/benchmarks.py.  
* Dataset sizes: MTEB-dataset-sizes.md, generated from local task metadata plus live HF Hub API responses.

# Running MTEB on BidirLM-Omni

**Running MTEB on BidirLM-Omni**  
This is the concrete runbook for evaluating \`BidirLM/BidirLM-Omni-2.5B-Embedding\` (https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding) with MTEB. Use MTEB.md for the general benchmark process.

**Model facts**

| Item | Value |
| :---- | :---- |
| Main MTEB entry point | mteb.get\_model("BidirLM/BidirLM-Omni-2.5B-Embedding") |
| Local MTEB registration | Present in mteb/mteb/models/model\_implementations/bidirlm\_omni\_models.py |
| Parameters / embedding size | 2,445,009,536 params; 2048-d embeddings |
| Local MTEB memory metadata | memory\_usage\_mb=4663 |
| Local MTEB modalities | \["text", "image", "audio", "video"\] |
| HF model-card modalities | Text, image, and audio embeddings in one 2048-d shared space |
| HF repo size | About 4.91 GB total; model.safetensors about 4.89 GB |
| Model requirements | transformers\>=5.5.0, sentence-transformers\>=5.4.0, librosa\>=0.10.0, trust\_remote\_code=True, cuDNN \> 9.20.0 |

Important distinction: the HF model card advertises text/image/audio embedding use. The local MTEB clone registers a video-enabled wrapper too, using MTEB video/audio collators. Treat video results as local-MTEB-wrapper behavior that should be smoke-tested before reporting.

**Run command**

pip install "mteb\[image,audio,video\]"

pip install "transformers\>=5.5.0" "sentence-transformers\>=5.4.0" "librosa\>=0.10.0"

Install caveats:

* mteb\[audio\] may require FFmpeg and recent Transformers when used with datasets\>=4.  
* mteb\[video\] pulls video decoding support.  
* The model uses remote custom code, so loading must allow trust\_remote\_code=True; MTEB's registered loader does this.  
* The model card recommends cuDNN \> 9.20.0 because earlier versions can trigger a Conv3D slowdown.

Recommended smoke test:

import mteb

model \= mteb.get\_model("BidirLM/BidirLM-Omni-2.5B-Embedding")

tasks \= mteb.get\_tasks(tasks=\["ArguAna"\])

results \= mteb.evaluate(

    model,

    tasks=tasks,

    encode\_kwargs={"batch\_size": 8},

)

CLI equivalent:

mteb run \\

  \-m BidirLM/BidirLM-Omni-2.5B-Embedding \\

  \-t ArguAna \\

  \--output-folder results

After the smoke test, measure actual disk use:

du \-sh "${HF\_HOME:-$HOME/.cache/huggingface}" \~/.cache/mteb results 2\>/dev/null

**What our team must provide**  
For this exact model, no new MTEB model definition is needed in the current local clone. We only choose tasks or benchmarks and provide compute.

Suggested task progression:

| Goal | Benchmark/task | Why |
| :---- | :---- | :---- |
| Fast integration check | ArguAna | One text retrieval task; validates model loading, encoding, scoring, and result output. |
| English text benchmark | mteb.get\_benchmark("MTEB(eng, v2)") | Main English text suite. Retrieval tasks dominate runtime. |
| Multilingual text benchmark | mteb.get\_benchmark("MTEB(Multilingual, v2)") | Broader language coverage, higher runtime and disk use. |
| Image/image-text benchmark | mteb.get\_benchmark("MIEB(lite)") | 51-task image benchmark subset. |
| Audio/audio-text benchmark | mteb.get\_benchmark("MAEB(beta)") | 30-task beta audio benchmark. |
| Audio-only pilot | mteb.get\_benchmark("MAEB(beta, audio-only)") | Smaller audio-only subset for isolating audio path issues. |

**What MTEB provides**  
The registered ModelMeta points to BidirLMOmniEncoder, sets trust\_remote\_code=True, declares the model revision, embedding dimension, memory metadata, and modalities, and uses per-task instruction prompts. The encoder implements one encode(...) path, detects which modality columns are present, and routes audio/video through MTEB collators.

For audio, the local wrapper resamples to 16 kHz and caps audio at about 30 seconds. For video, it samples at about 2 fps with a 64-frame cap by default. Reduce encode\_kwargs\["batch\_size"\] first if GPU memory is tight.

**Compute and risks**  
A single small text task should run in minutes once dependencies and weights are cached. Full text, image, or audio suites should be treated as multi-hour jobs on one GPU; multilingual and retrieval-heavy suites can run much longer because corpus encoding dominates.

Use a 16-24 GB GPU as a practical starting point for smoke tests. Larger batches, multimodal tasks, and full-suite runs may need more memory. The model weights alone are roughly 4.9 GB on disk, while evaluation data size depends entirely on selected benchmarks.

Open risks before reporting numbers:

* We have source-checked registration and benchmark names, but a live MTEB run was not completed in this environment. The current shell Python is 3.9.21, while the local MTEB clone declares Python \>=3.10.  
* Video support is declared in the local MTEB wrapper but is not the main modality advertised on the HF model card, so video results need extra validation.  
* Published runtime claims should come from our own pilot run with recorded GPU type, batch size, wall time, and cache size.

Sources: MTEB install (https://embeddings-benchmark.github.io/mteb/installation/), running evaluations (https://embeddings-benchmark.github.io/mteb/get\_started/usage/running\_the\_evaluation/), defining models (https://embeddings-benchmark.github.io/mteb/get\_started/usage/defining\_the\_model/), adding models (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_model/), adding tasks (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_dataset/), adding benchmarks (https://embeddings-benchmark.github.io/mteb/contributing/adding\_a\_benchmark/), available benchmarks (https://embeddings-benchmark.github.io/mteb/overview/available\_benchmarks/), BidirLM model card (https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding).

See also: MTEB general runbook (MTEB.md).

# MTEB / MIEB / MAEB model coverage

**MTEB / MIEB / MAEB model coverage**  
**Question:** which multimodal/omni embedding models does MTEB already support, and which recent ones are still missing?

**Verification status:** refreshed against the local benchmarks/mteb/ clone, which matches upstream embeddings-benchmark/mteb HEAD e6980ebe from 2026-05-26. Public docs also list the current MIEB and MAEB benchmark families.

**Method:** parsed all ModelMeta(...) calls in mteb/mteb/models/model\_implementations/, including shared \*\*\_COMMON kwargs, then checked missing-model claims with case-insensitive grep over mteb/mteb/models/. The framework modality set is text, image, audio, video.

**Coverage snapshot**

* **716 total model registrations** across 192 implementation files.  
* **221 registrations declare at least one non-text modality**; excluding the two random baseline registrations, that is **219 real multimodal/audio/image/video model entries**.  
* Main modality groups, excluding random baselines:

| Modality set | Count | Representative entries |
| :---- | :---- | :---- |
| image+text | 105 | CLIP/OpenCLIP, SigLIP v1, EVA02-CLIP, MetaCLIP 2 b32, BLIP/BLIP2, GME-Qwen2-VL, VLM2Vec, Qwen3-VL-Embedding, Cohere embed-v4, Voyage multimodal-3, visual-document retrievers |
| audio only | 42 | wav2vec2, HuBERT, WavLM, Whisper, AST, VGGish, YAMNet, EnCodec, MMS, data2vec, SEW-D, SeamlessM4T-v2, SpeechT5 ASR, UniSpeech-SAT, MCTCT, CNN14, HeAR |
| audio+text | 11 | CLAP variants, MSCLAP, MuQ-MuLan, Qwen2-Audio, wav2clip, SpeechT5 multimodal |
| image only | 22 | DINOv2, Web-SSL DINO/MAE, MoCo-v3, Penguin-Encoder |
| video only | 8 | V-JEPA2 variants |
| text+video | 3 | Microsoft XCLIP variants |
| image+text+video | 4 | VLM2Vec-V2, ebind-points-vision, UME-R1 2B/7B |
| audio+text+video | 6 | Meta PE-AV small/base/large and 16-frame variants |
| audio+image+text | 1 | Gemini embedding 2 preview |
| text+image+audio+video | 17 | BidirLM-Omni, e5-omni, LCO-Omni, Qwen2.5/Qwen3-Omni, OmniEmbed, eBind, jina v5 omni, omni-embed-nemotron, OmniVinci |

**Already covered**  
**Image+text is the mature area.** MTEB already includes classic CLIP/OpenCLIP/SigLIP v1/EVA02-CLIP models, BLIP/BLIP2, Jina CLIP and Jina v4, GME-Qwen2-VL, VLM2Vec, mmE5, e5-v, Qwen3-VL-Embedding, Seed1.6, Cohere multimodal embeddings, Voyage multimodal-3, Nomic multimodal models, and many visual-document retrieval families such as ColPali, ColQwen, ColSmol, ColNomic, ModernVBERT, VisRAG-Ret, Nemotron VL, Argus, Ops, SauerkrautLM, NanoVDR, Granite Vision, and webAI ColVec.

**Audio coverage is now substantial.** MAEB support means the registered audio/audio-text models are not just catalog entries; they have a dedicated benchmark family. The docs and local registry expose MAEB(beta) with 30 tasks, MAEB(beta, audio-only) with 19 tasks, MAEB(beta, extended) with 89 tasks, and MAEB+(beta) with 98 tasks.

**Omni and video coverage is broader than the old note said.** In addition to the 17 full text+image+audio+video models, the clone includes PE-AV for audio/video/text, XCLIP for text/video, UME-R1 and VLM2Vec-V2 for image/text/video, and V-JEPA2 video-only variants.

**Evaluation suites**

* **MIEB** exists for image/image-text evaluation: MIEB(eng) has 125 tasks, MIEB(Multilingual) has 130 tasks across 39 languages, MIEB(lite) has 51 tasks, and MIEB(Img) has 49 image-only tasks.  
* **MAEB** exists for audio/audio-text evaluation: see the 30/19/89/98-task variants above.  
* **Video gap:** local task metadata defines MVEB-style task types and the repo contains many video/audio-video tasks, and as of 2026-06-19 the local benchmark registry now defines **MVEB(beta)** (23 tasks), **MVEB(video, beta)** (9), and **MVEB(text, video, beta)** (19) in `benchmarks.py` — the earlier "no MVEB benchmark" gap is closed. Remaining work is running models on these suites, not packaging them.

**Missing or partial gaps**  
These were verified absent as real local ModelMeta registrations. Some names appear only in comments or as backbones inside another wrapper, which is not the same as a directly runnable MTEB model.

| Model/family | Org | Approx. release | Modalities | Status / why it matters | Add effort |
| :---- | :---- | :---- | :---- | :---- | :---- |
| SigLIP 2 | Google | 2025 | image+text | Successor to registered SigLIP v1 family | Easy: extend siglip\_models.py |
| MobileCLIP / MobileCLIP2 | Apple | 2024/2025 | image+text | Efficient edge CLIP family | Easy/medium via OpenCLIP-style wrapper if HF/open\_clip checkpoints fit |
| EVA-CLIP-8B / EVA-CLIP-18B | BAAI | 2024 | image+text | Larger EVA-CLIP models; current registry has EVA02 variants | Easy/medium |
| DFN CLIP | Apple | 2023/2024 | image+text | Strong Data Filtering Network CLIP checkpoints; only a comment hit exists locally | Easy if available through OpenCLIP |
| Larger MetaCLIP 2 variants | Meta | 2025 | image+text | Only facebook/metaclip-2-mt5-worldwide-b32 is registered | Easy: extend metaclip\_models.py |
| BGE-VL | BAAI | 2024/2025 | image+text | Strong visual-language embedding/retrieval family | Medium: likely new wrapper |
| MM-Embed | NVIDIA | 2024 | image+text | Universal multimodal retrieval model cited around M-BEIR/UniIR | Medium |
| LLaVE | XMU | 2025 | image+text | VLM embedding model family | Medium |
| LamRA | \- | 2025 | image+text | Qwen2-VL retrieval assistant style | Medium |
| UniIR / M-BEIR reference models | TIGER-Lab | 2024 | image+text | Important universal multimodal retrieval baselines | Medium |
| MagicLens | Google | 2024 | image+text | Instruction-conditioned image retrieval; availability/licensing may block | Hard |
| ImageBind as a standalone model | Meta | 2023 | text+image+audio+depth/IMU/etc. | Currently only appears inside eBind comments/backbone logic | Medium |
| LanguageBind | PKU/Yuxin team | 2024 | text+image+audio+video+depth | Canonical language-anchored multimodal binder | Medium |
| OmniBind | \- | 2024 | text+image+audio+3D | Broad multi-space binding family | Hard |
| AudioCLIP | \- | 2021/2022 | audio+image+text | Early audio/image/text embedding baseline | Medium |
| Voyage multimodal-3.5 | Voyage AI | 2026 | text+image/video API | Newer API tier; voyage-multimodal-3 is registered | Easy if API surface is compatible |

Not missing anymore / should not be listed as gaps: Qwen3-VL-Embedding, Jina v4 and v5 omni, GME-Qwen2-VL, VLM2Vec and VLM2Vec-V2, Cohere embed-v4, Gemini embedding 2 preview, OmniEmbed, OmniVinci, e5-omni, LCO-Omni, BidirLM-Omni, PE-AV, UME-R1, and MAEB.

**Recommendation**

1. **Quick wins:** add SigLIP 2, larger MetaCLIP 2 variants, and Voyage multimodal-3.5 if their loader/API surfaces match the existing wrappers.  
2. **High-value omni gaps:** add standalone ImageBind and LanguageBind wrappers; both are more central to omni embedding coverage than another CLIP variant.  
3. **Retrieval-SOTA gaps:** prioritize BGE-VL, MM-Embed, LLaVE, LamRA, and UniIR/M-BEIR reference models if the goal is visual retrieval coverage.  
4. **Benchmark packaging gap:** if the team wants to evaluate omni/video embeddings, the missing upstream contribution is a named video/omni benchmark built from existing video/audio-video tasks.

Sources checked: local upstream-matching MTEB clone at e6980ebe, MTEB available benchmarks (https://embeddings-benchmark.github.io/mteb/overview/available\_benchmarks/), MTEB multimodal models (https://embeddings-benchmark.github.io/mteb/overview/available\_models/multimodal/), MTEB image-text models (https://embeddings-benchmark.github.io/mteb/overview/available\_models/image\_text/), MTEB audio models (https://embeddings-benchmark.github.io/mteb/overview/available\_models/audio/), MTEB audio-text models (https://embeddings-benchmark.github.io/mteb/overview/available\_models/audio\_text/), and the MAEB arXiv entry linked from MTEB. External spot checks for missing-family claims: Voyage multimodal embeddings (https://docs.voyageai.com/docs/multimodal-embeddings), SigLIP 2 (https://huggingface.co/blog/siglip2), BGE-VL (https://bge-model.com/bge/bge\_vl.html), MobileCLIP2 (https://machinelearning.apple.com/research/mobileclip2), MM-Embed (https://huggingface.co/nvidia/MM-Embed), MagicLens (https://arxiv.org/abs/2403.19651), UniIR (https://tiger-ai-lab.github.io/UniIR/), LanguageBind (https://huggingface.co/LanguageBind/LanguageBind\_Audio\_FT), and OmniBind (https://omnibind.github.io/).

