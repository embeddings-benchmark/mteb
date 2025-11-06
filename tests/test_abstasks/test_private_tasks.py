from mteb.get_tasks import get_tasks

# List of accepted private tasks - update this list as needed
ACCEPTED_PRIVATE_TASKS = [
    "JapaneseCode1Retrieval",
    "Code1Retrieval",
    "EnglishFinance1Retrieval",
    "EnglishFinance2Retrieval",
    "EnglishFinance3Retrieval",
    "EnglishFinance4Retrieval",
    "EnglishHealthcare1Retrieval",
    "French1Retrieval",
    "FrenchLegal1Retrieval",
    "German1Retrieval",
    "GermanHealthcare1Retrieval",
    "GermanLegal1Retrieval",
    "JapaneseLegal1Retrieval",
    "Vidore3TelecomRetrieval",
    "Vidore3NuclearRetrieval",
    # Add task names here that are allowed to be private
]


def test_private_tasks_fail_unless_accepted():
    """Test that private tasks (is_public=False) fail unless they are in the accepted list."""
    # Get all tasks including private ones
    all_tasks = get_tasks(exclude_private=False)

    # Find all private tasks
    private_tasks = [task for task in all_tasks if task.metadata.is_public is False]

    # Check that all private tasks are in the accepted list
    for task in private_tasks:
        assert task.metadata.name in ACCEPTED_PRIVATE_TASKS, (
            f"Private task '{task.metadata.name}' is not in the accepted private tasks list. "
            f"Either make the task public (is_public=True) or add it to ACCEPTED_PRIVATE_TASKS."
        )
