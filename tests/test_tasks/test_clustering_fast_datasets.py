from __future__ import annotations

import pytest

from mteb.abstasks.text.abs_text_clustering_fast import AbsTextClusteringFast


@pytest.mark.parametrize("dataset", AbsTextClusteringFast.__subclasses__())
def test_clustering_fast_datasets(dataset):
    assert (
        dataset.max_document_to_embed is None
        and dataset.max_fraction_of_documents_to_embed > 0
    ) or (
        dataset.max_document_to_embed > 0
        and dataset.max_fraction_of_documents_to_embed is None
    )
