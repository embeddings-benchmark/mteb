from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskAbstention import AbsTaskAbstention
from ...Reranking.fra.AlloprofReranking import AlloprofReranking
from ....evaluation.evaluators import AbstentionRerankingEvaluator


class AlloprofRerankingAbstention(AbsTaskAbstention, AlloprofReranking):
    metadata = TaskMetadata(
        name="AlloprofRerankingAbstention",
        description="This dataset was provided by AlloProf, an organisation in Quebec, Canada offering resources and a help forum curated by a large number of teachers to students on all subjects taught from in primary and secondary school",
        reference="https://huggingface.co/datasets/antoinelb7/alloprof",
        dataset={
            "path": "lyon-nlp/mteb-fr-reranking-alloprof-s2p",
            "revision": "e40c8a63ce02da43200eccb5b0846fcaa888f562",
        },
        type="Abstention",
        category="s2s",
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
