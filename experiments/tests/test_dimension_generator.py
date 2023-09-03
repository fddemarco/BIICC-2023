import pytest
import pandas as pd
from pandas import testing as tm

from experiments.dimension_generator import (
    DimensionGenerator,
    WallerDimenGenerator,
    score_embedding,
)


@pytest.fixture(params=[
    ([1, 0, 0], [-1, 0, 0]),
    ([1, 0], [0, 1]),
    ([1, 2, 3, 4, 5], [0, 1, -1, 2, -2])
])
def embeddings(request):
    subreddits = ["Conservative", "democrats"]
    data = [*request.param]
    return pd.DataFrame(data, index=subreddits)


@pytest.fixture()
def seeds():
    return [("Conservative", "democrats")]


@pytest.fixture()
def dimen_names():
    return ["dem_rep"]


@pytest.fixture()
def scores(embeddings, seeds, dimen_names):
    generator = DimensionGenerator(embeddings)
    return generator.get_scores_from_seeds(seeds, dimen_names)


@pytest.fixture()
def waller_scores(embeddings, seeds, dimen_names):
    dimen_generator = WallerDimenGenerator(embeddings)
    return dimen_generator.get_scores(seeds, dimen_names)


class TestDimensionGenerator:
    def test_01(self, scores, waller_scores):
        tm.assert_series_equal(scores.dem_rep, waller_scores.dem_rep)

    def test_02(self, scores):
        dummy_series = scores.dem_rep == scores.dem_rep
        values_in_range = (scores.dem_rep <= 1) & (scores.dem_rep >= -1)
        tm.assert_series_equal(dummy_series, values_in_range)
