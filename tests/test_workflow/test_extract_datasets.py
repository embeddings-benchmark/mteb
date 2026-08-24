from unittest.mock import Mock, patch

from scripts import extract_datasets as extract_datasets_module


def test_extract_added_datasets_only_uses_added_files(capsys):
    changed_files = ["mteb/tasks/added.py", "mteb/tasks/modified.py"]
    with (
        patch.object(
            extract_datasets_module, "get_changed_files", return_value=changed_files
        ),
        patch.object(extract_datasets_module, "Repo") as repo_class,
        patch.object(
            extract_datasets_module, "extract_datasets", return_value=[]
        ) as extract,
    ):
        repo = repo_class.return_value
        base, head = Mock(), Mock()
        repo.commit.side_effect = [base, head]
        base.diff.return_value = [Mock(b_path=changed_files[0])]
        extract_datasets_module.extract_added_datasets("base")

    base.diff.assert_called_once_with(head, diff_filter="A")
    extract.assert_called_once_with([changed_files[0]])
    assert extract_datasets_module.extract_datasets([]) == []
    assert capsys.readouterr().out == 'export CUSTOM_DATASET_REVISIONS="__EMPTY__"\n'
