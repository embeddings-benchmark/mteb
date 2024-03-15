from ....abstasks.AbsTaskReranking import AbsTaskReranking


class StackOverflowDupQuestions(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "StackOverflowDupQuestions",
            "hf_hub_name": "mteb/stackoverflowdupquestions-reranking",
            "description": (
                "Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python"
            ),
            "reference": "https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "map",
            "revision": "e185fbe320c72810689fc5848eb6114e1ef5ec69",
        }
