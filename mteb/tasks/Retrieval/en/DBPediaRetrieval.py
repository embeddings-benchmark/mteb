from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class DBPedia(AbsTaskRetrieval):
    metadata = TaskMetadata()

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)
        return {
            "name": "DBPedia",
            "hf_hub_name": "mteb/dbpedia",
            "description": (
                "DBpedia-Entity is a standard test collection for entity search over the DBpedia knowledge base"
            ),
            "reference": "https://github.com/iai-group/DBpedia-Entity/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev", "test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
            "revision": "c0f706b76e590d620bd6618b3ca8efdd34e2d659",
        }
