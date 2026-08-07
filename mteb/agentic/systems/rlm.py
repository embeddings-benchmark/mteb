"""Wrap the official Recursive Language Model (RLM) library as an AnswerSystem.

RLM (github.com/alexzhang13/rlm, arXiv 2512.24601) offloads the context into a
REPL variable and lets the model write code to examine and recursively query
it. This adapter builds the context from the corpus and runs rlm.completion.
It uses rlm's default in-process "local" environment (just pip install rlm, no
Docker); pass environment="docker" (with a sandbox image) or a cloud backend
(e2b, daytona, modal) for isolated execution of the model-written code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mteb.agentic.interface import AnswerResult, Usage
from mteb.agentic.systems._common import join_all

if TYPE_CHECKING:
    from mteb.agentic.interface import ChatModel, CorpusHandle

# The question goes in the root prompt (seen by the root LM); only the documents
# are offloaded into the REPL variable, else the root LM cannot see the task.
_ROOT_PROMPT = (
    "Answer this question using the documents offloaded to the context variable. "
    "Search them (grep/filter/scan) and return ONLY the final short answer.\n\n"
    "Question: {question}"
)


class RLMSystem:
    """Recursive Language Model over the raw corpus, via the official rlm library.

    RLM has no first-stage retriever: the whole corpus is loaded as the context
    and the model greps it with code, so no retriever/CorpusHandle.search is used.
    max_context_chars bounds the context for feasibility. environment selects
    rlm's execution backend: "local" (default, in-process, no Docker) runs the
    model-written code in this process; "docker" (needs the sandbox_image) or a
    cloud backend (e2b, daytona, modal) isolates it.
    """

    def __init__(
        self,
        model: ChatModel,
        *,
        max_context_chars: int = 2_000_000,
        max_depth: int = 1,
        max_iterations: int = 30,
        environment: str = "local",
        sandbox_image: str = "rlm-sandbox",
        timeout_s: float = 90.0,
    ) -> None:
        self.model_name = model.name
        self.base_url = model.base_url
        self.api_key = model.api_key or "EMPTY"
        self.max_context_chars = max_context_chars
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.environment = environment
        self.sandbox_image = sandbox_image
        self.timeout_s = timeout_s
        self.name = f"rlm/{model.name}"

    def _build_rlm(self) -> Any:
        try:
            from rlm import RLM  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RLMSystem needs the rlm library (imported as rlm, PyPI package "
                'rlms). Install it with pip install "mteb[agentic-rlm]".'
            ) from exc
        # Only the docker backend needs an image; local/cloud backends do not.
        env_kwargs = (
            {"image": self.sandbox_image} if self.environment == "docker" else {}
        )
        return RLM(
            backend="openai",
            backend_kwargs={
                "model_name": self.model_name,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "timeout": self.timeout_s,
            },
            environment=self.environment,
            environment_kwargs=env_kwargs,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
        )

    def answer(self, question: str, corpus: CorpusHandle) -> AnswerResult:
        """Load the whole corpus as context, then let RLM search it for the answer."""
        context = join_all(corpus)
        out = self._build_rlm().completion(
            context[: self.max_context_chars],
            root_prompt=_ROOT_PROMPT.format(question=question),
        )
        # The library does not expose per-call usage; latency is filled by the evaluator.
        return AnswerResult(answer=out.response, usage=Usage())
