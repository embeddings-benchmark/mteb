from __future__ import annotations

from pathlib import Path

from mteb.get_tasks import get_task


def test_worldsense_task_registered() -> None:
    task_cls = get_task("WorldSense1MinDomainClustering")
    assert task_cls.metadata.name == "WorldSense1MinDomainClustering"
    assert task_cls.metadata.type == "VideoClustering"


def test_worldsense_descriptive_stats_on_disk() -> None:
    task_cls = get_task("WorldSense1MinDomainClustering")
    stats = task_cls.metadata.descriptive_stats
    assert stats is not None
    assert stats["test"]["num_samples"] == 568
    assert stats["test"]["labels_statistics"]["unique_labels"] == 8

    path = Path(__file__).resolve().parents[2] / "mteb" / "descriptive_stats" / "VideoClustering" / "WorldSense1MinDomainClustering.json"
    assert path.is_file()
