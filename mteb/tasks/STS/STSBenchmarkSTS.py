from ...abstasks.AbsTaskSTS import AbsTaskSTS


class STSBenchmarkSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STSBenchmark",
            "hf_hub_name": "mteb/stsbenchmark-sts",
            "description": "Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
            "reference": "http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark",
            "type": "STS",
            "category": "s2s",
            "available_splits": ["train", "validation", "test"],
            "available_langs": ["en"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
        }
