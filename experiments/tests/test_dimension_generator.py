import pytest
import pandas as pd
from pandas import testing as tm

from experiments.dimension_generator import (
    DimensionGenerator,
    WallerDimenGenerator,
    score_embedding,
)


@pytest.fixture(
    params=[
        [
            [0.76519959, 0.03239545, 0.53424408, 0.44266574, 0.55727791],
            [0.32894575, 0.86130903, 0.34940292, 0.04351816, 0.64656009],
            [0.25120826, 0.90388639, 0.20334958, 0.59713476, 0.21468545],
            [0.56375595, 0.58099063, 0.08095738, 0.14105793, 0.0450353],
            [0.41163704, 0.29081786, 0.81509444, 0.071807, 0.96327111],
        ],
        [
            [0.67642666, 0.29096151, 0.37820934],
            [0.27031607, 0.70738351, 0.17957367],
            [0.19664239, 0.12260562, 0.74576265],
            [0.20030666, 0.75673594, 0.34827141],
            [0.90427689, 0.36045301, 0.67793955],
            [0.15285038, 0.63472716, 0.66028855],
            [0.47601083, 0.21928615, 0.17682311],
            [0.58327315, 0.60013272, 0.22903184],
            [0.28489212, 0.5841554, 0.93369358],
            [0.09231587, 0.60331412, 0.89910045],
            [0.67490817, 0.07553023, 0.09609613],
            [0.05429499, 0.33976268, 0.78866882],
            [0.13013662, 0.94582214, 0.68290145],
            [0.78659041, 0.46457285, 0.8901706],
            [0.75863147, 0.10696646, 0.15007107],
            [0.9905003, 0.85784344, 0.69245329],
            [0.60721303, 0.50883191, 0.30438918],
            [0.06913729, 0.925505, 0.8609644],
            [0.86421834, 0.13436222, 0.68869372],
            [0.2763716, 0.22057208, 0.64824115],
        ],
        [[0.19215367, 0.07447659, 0.34677997, 0.61999938],
       [0.85769497, 0.47007512, 0.5157554 , 0.82719861],
       [0.27367439, 0.63112969, 0.22969539, 0.36045082],
       [0.24980312, 0.63982306, 0.3957246 , 0.83138228],
       [0.04923169, 0.79540832, 0.43595351, 0.75005156],
       [0.29421704, 0.60441345, 0.94073198, 0.0629855 ],
       [0.18569706, 0.10222294, 0.88576099, 0.85773229],
       [0.90366641, 0.60486249, 0.76656001, 0.88252291],
       [0.58484002, 0.52925899, 0.88195509, 0.06965376],
       [0.14643428, 0.62119783, 0.76216916, 0.95629683],
       [0.04106485, 0.99918406, 0.73537739, 0.81619452],
       [0.39920669, 0.74431558, 0.87749139, 0.39754929],
       [0.45358082, 0.01967785, 0.96484697, 0.91349083],
       [0.42911647, 0.41932921, 0.23221466, 0.81632824],
       [0.22812161, 0.81684781, 0.55631248, 0.59865408]]
    ]
)
def data(request):
    return request.param


@pytest.fixture
def index(data):
    return [f"index:{i}" for i in range(len(data))]


@pytest.fixture
def embeddings(data, index):
    return pd.DataFrame(data, index=index)


@pytest.fixture()
def seeds(index):
    return [(index[0], index[-1])]


@pytest.fixture()
def dimen_names():
    return ["dem_rep"]


@pytest.fixture()
def scores(embeddings, seeds, dimen_names):
    generator = DimensionGenerator(embeddings)
    return generator.get_scores_from_seeds(seeds, dimen_names).dem_rep


@pytest.fixture()
def waller_scores(embeddings, seeds, dimen_names):
    dimen_generator = WallerDimenGenerator(embeddings)
    return dimen_generator.get_scores(seeds, dimen_names).dem_rep


class TestDimensionGenerator:
    def test_01(self, scores, waller_scores):
        tm.assert_series_equal(scores, waller_scores)

    def test_02(self, scores):
        values_in_range = (scores <= 1) & (scores >= -1)
        assert values_in_range.all()
