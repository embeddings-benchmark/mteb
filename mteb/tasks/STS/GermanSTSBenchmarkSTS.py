from ...abstasks.AbsTaskSTS import AbsTaskSTS


class GermanSTSBenchmarkSTS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "GermanSTSBenchmark",
            "hf_hub_name": "jinaai/german-STSbenchmark",
            "description": "Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. "
                           "Translations were originally done by T-Systems on site services GmbH.",
            "reference": "https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["de"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            "revision": "49d9b423b996fea62b483f9ee6dfb5ec233515ca",
        }
