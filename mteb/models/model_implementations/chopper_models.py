"""Chopper: a reranker and context compressor in one forward pass (https://github.com/wiiiktor/chopper_demo).

Chopper reads a question together with ALL its candidate passages in one input (up to 8,192 tokens; longer inputs are
packed into several passes) and predicts, per token, whether it belongs to a quote that answers the question. A passage's
relevance is the mean p(KEEP) of its 16 most likely tokens. MTEB hands cross-encoders flat (query, document) pairs, so
the wrapper groups the pairs by query and scores each group with one `Chopper.run` call.

Long documents are ranked by their best window: documents longer than `window_chars` (900) are cut into windows of that
size (stride half a window), every window is scored, and then each document's best window is scored together with the
other candidates in one comparison. The whole text is read, and the final scores are comparable across candidates
(long documents spread over several separate reads otherwise compare poorly: AlloprofReranking 0.571 -> 0.737).
Pass `window_chars=None` to score full documents directly.

A passage's relevance is relative to the other passages in the same read, so scores from separate reads compare poorly.
Long candidate lists are therefore ranked in rounds: blocks of `block` (50) candidates in the given order, the best `keep`
(25) of each block read together, then the best `reread` (10) of those read together once more (HotpotQAHardNegatives,
top 100 of a first stage: NDCG@10 0.835 in one ranking vs 0.876 in rounds). Up to `block` candidates this is one plain read;
`block=None, reread=None` gives one plain ranking.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.models.model_meta import ModelMeta

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

CHOPPER_LANGUAGES = [
    "eng-Latn",
    "pol-Latn",
    "deu-Latn",
    "fra-Latn",
    "spa-Latn",
    "ita-Latn",
    "por-Latn",
    "nld-Latn",
    "rus-Cyrl",
    "ukr-Cyrl",
    "ces-Latn",
    "zho-Hans",
    "jpn-Jpan",
    "kor-Hang",
    "ara-Arab",
    "hin-Deva",
]

# Training data (all train splits): MuSiQue, HotpotQA, TriviaQA, 2WikiMultiHopQA, MuSiQue-unanswerable, our own
# Wikipedia (20231101.en) 1-2 hop questions, multilingual web rerank pools (FineWeb, 8 languages, LLM-labelled),
# QASPER, WildChat and Multi-Session Chat conversations, PoQuAD and PolQA. Only HotpotQA is an MTEB task.
CHOPPER_TRAINING_DATA = {"HotpotQA", "HotpotQAHardNegatives"}


class ChopperWrapper:
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        variant: str = "base",
        window_chars: int | None = 900,
        block: int | None = 50,
        keep: int = 25,
        reread: int | None = 10,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Chopper as a reranker.

        Args:
            model_name: Hugging Face repository of the Chopper weights.
            revision: Repository revision (unused; the weights are loaded by `chopper_rag`).
            variant: Encoder variant inside the repository ("base" or "small").
            window_chars: Documents longer than this many characters are split into half-overlapping windows; a
                document is represented by its best window. None reads every document whole.
            block: Candidate lists longer than this are ranked in rounds: blocks of `block` candidates are read
                separately and the best `keep` of each block go on to the next round. None reads the whole list in
                one pass. Must be at least 2.
            keep: How many candidates of each block go on to the next round. Must be at least 1 and smaller than
                `block`, so that every round shortens the list.
            reread: After a block round, the best `reread` survivors are read together once more. None or 0
                disables the re-read.
            device: Torch device; "auto" (the default) picks CUDA when available.
            **kwargs: Ignored.
        """
        if window_chars is not None and window_chars < 1:
            raise ValueError(
                f"window_chars must be positive or None, got {window_chars}"
            )
        if block is not None:
            if block < 2:
                raise ValueError(f"block must be at least 2 or None, got {block}")
            if not 1 <= keep < block:
                raise ValueError(
                    f"keep must satisfy 1 <= keep < block ({block}), got {keep}"
                )
        if reread is not None and reread < 0:
            raise ValueError(f"reread must be non-negative or None, got {reread}")
        from chopper_rag import Chopper

        self.model = Chopper.from_pretrained(
            repo=model_name, variant=variant, device=device or "auto"
        )
        self.window_chars = window_chars
        self.block, self.keep, self.reread = block, keep, reread

    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = [text for batch in inputs1 for text in batch["text"]]
        documents = [text for batch in inputs2 for text in batch["text"]]
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for i, query in enumerate(queries):
            groups.setdefault(query, []).append(i)
        scores = np.zeros(len(queries), dtype=np.float32)
        for query, idx in groups.items():
            scores[idx] = self._score(query, [documents[i] for i in idx])
        return scores

    def _windows(self, text: str) -> list[str]:
        w = self.window_chars
        if not w or len(text) <= w:
            return [text]
        step = w // 2
        return [text[i : i + w] for i in range(0, len(text) - step, step)]

    def _score(self, query: str, docs: list[str]) -> np.ndarray:
        return self._tournament(query, docs)

    def _read(self, query: str, docs: list[str]) -> np.ndarray:
        """one comparison of `docs`: long documents by their best window, chosen among the windows of these docs"""
        windows = [self._windows(d) for d in docs]
        if all(len(w) == 1 for w in windows):
            return np.asarray(self.model.run(query, docs).passage_scores)
        flat = [x for w in windows for x in w]
        flat_scores = self.model.run(query, flat).passage_scores
        best, k = [], 0
        for w in windows:
            best.append(w[int(np.argmax(flat_scores[k : k + len(w)]))])
            k += len(w)
        return np.asarray(self.model.run(query, best).passage_scores)

    def _tournament(self, query: str, docs: list[str]) -> np.ndarray:
        """Chopper's scores are relative to the passages read in the same pass, so a long candidate list is ranked
        in rounds. While more than `block` candidates remain: read them in blocks of `block` (in the given order) and
        keep the best `keep` of each block. Then read the survivors together and, after a block round, read the best
        `reread` of them together once more. A list of up to `block` candidates gets one plain read. A candidate's
        score is (rounds survived) + its p(KEEP) score in the last pass it was in, so every survivor ranks above
        every candidate dropped earlier."""
        out = np.zeros(len(docs), dtype=np.float64)
        cur = list(range(len(docs)))

        def read(idx: list[int], level: int) -> dict[int, float]:
            s = self._read(query, [docs[i] for i in idx])
            for i, v in zip(idx, s, strict=True):
                out[i] = level + float(v)
            return dict(zip(idx, s, strict=True))

        level = 0
        while self.block and len(cur) > self.block:
            nxt = []
            for b in range(0, len(cur), self.block):
                sc = read(cur[b : b + self.block], level)
                nxt += sorted(sc, key=lambda i: -sc[i])[: self.keep]
            cur = sorted(nxt)
            level += 1
        sc = read(cur, level)
        if self.reread and level > 0 and len(cur) > self.reread:
            read(sorted(sorted(sc, key=lambda i: -sc[i])[: self.reread]), level + 1)
        return out


chopper_031b = ModelMeta(
    extra_requirements_groups=["chopper"],
    loader=ChopperWrapper,
    name="wiiiktor/chopper-0.31B",
    revision="d551e53419da0169ebfea7b1eb99177c788c4f2d",
    release_date="2026-11-10",
    languages=CHOPPER_LANGUAGES,
    n_parameters=306_941_955,
    n_embedding_parameters=196_608_000,
    memory_usage_mb=1200,
    max_tokens=8192,
    embed_dim=None,
    license="mit",
    open_weights=True,
    public_training_code="https://github.com/wiiiktor/chopper_demo",
    public_training_data="https://huggingface.co/datasets/wiiiktor/chopper_demo",
    framework=["PyTorch"],
    reference="https://huggingface.co/wiiiktor/chopper-0.31B",
    similarity_fn_name=None,
    use_instructions=False,
    training_datasets=CHOPPER_TRAINING_DATA,
    adapted_from="jhu-clsp/mmBERT-base",
    model_type=["cross-encoder"],
)
