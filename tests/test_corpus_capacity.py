from __future__ import annotations

from collections import Counter

import pytest


@pytest.mark.parametrize("language", ["en", "sv"])
def test_weighted_corpus_replenishes_beyond_initial_thousand(language):
    from nc_factory.text import DiversityIndex, generate_texts

    index, assigned, result = DiversityIndex(), Counter(), []
    for sequence in range(0, 1600, 32):
        result.extend(generate_texts(32, language, seed=100 + sequence, index=index, assigned=assigned))
    assert len(result) == 1600
    assert len({item.spoken_text for item in result}) == 1600
    assert len(assigned) == 17
    assert max(assigned.values()) - min(assigned.values()) <= 1
