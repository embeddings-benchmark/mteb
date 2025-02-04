from mteb.tasks.Reranking.eng.BIRCO.BIRCODorisMaeReranking import BIRCODorisMaeReranking
from mteb.tasks.Reranking.eng.BIRCO.BIRCOArguAnaReranking import BIRCOArguAnaReranking
from mteb.tasks.Reranking.eng.BIRCO.BIRCOClinicalTrialReranking import BIRCOClinicalTrialReranking
from mteb.tasks.Reranking.eng.BIRCO.BIRCORELICReranking import BIRCORELICReranking
from mteb.tasks.Reranking.eng.BIRCO.BIRCOWhatsThatBookReranking import BIRCOWhatsThatBookReranking

def test_birco_metadata():
    tasks = [
        BIRCODorisMaeReranking(),
        BIRCOArguAnaReranking(),
        BIRCOClinicalTrialReranking(),
        BIRCORELICReranking(),
        BIRCOWhatsThatBookReranking()
    ]
    for task in tasks:
        # Check that each task has a non-empty metadata name and description.
        assert task.metadata.name, "Metadata name should not be empty."
        assert task.metadata.description, "Metadata description should not be empty."
        # Check that get_query returns a string that starts with 'Instruction:'.
        sample = {"query": "Example query", "positive": ["doc1"], "negative": ["doc2"]}
        query_text = task.get_query(sample)
        assert query_text.startswith("Instruction:"), f"Query for {task.metadata.name} should start with 'Instruction:'"

if __name__ == '__main__':
    test_birco_metadata()
    print("All BIRCO task tests passed.")
