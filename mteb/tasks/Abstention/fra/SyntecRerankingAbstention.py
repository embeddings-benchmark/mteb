from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking
from ....evaluation.evaluators import AbstentionRerankingEvaluator


class SyntecRerankingAbstention(AbsTaskReranking):
    metadata = TaskMetadata(
        name="SyntecRerankingAbstention",
        description="This dataset has been built from the Syntec Collective bargaining agreement.",
        reference="https://huggingface.co/datasets/lyon-nlp/mteb-fr-reranking-syntec-s2p",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-syntec-s2p",
            "revision": "b205c5084a0934ce8af14338bf03feb19499c84d",
        },
        type="Abstention",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="map",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )


    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        scores = {}
        if self.is_multilingual:
            for lang in self.langs:
                data_split = self.dataset[lang][split]
                evaluator = AbstentionRerankingEvaluator(data_split, **kwargs)
                scores[lang] = evaluator(model)
        else:
            data_split = self.dataset[split]

            evaluator = AbstentionRerankingEvaluator(data_split, **kwargs)
            scores = evaluator(model)

        return dict(scores)