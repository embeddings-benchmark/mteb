from ....abstasks.AbsTaskClustering import AbsTaskClustering


class EightTagsClustering(AbsTaskClustering):
    @property
    def metadata_dict(self):
        return {
            "name": "8TagsClustering",
            "hf_hub_name": "PL-MTEB/8tags-clustering",
            "description": "Clustering of headlines from social media posts in Polish belonging to 8 categories: film, history, "
            "food, medicine, motorization, work, sport and technology.",
            "reference": "https://aclanthology.org/2020.lrec-1.207.pdf",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pl"],
            "main_score": "v_measure",
        }
