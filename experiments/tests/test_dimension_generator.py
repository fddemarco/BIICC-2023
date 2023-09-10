import pytest
import pandas as pd
from pandas import testing as tm
import numpy as np

from experiments.dimension_generator import (
    DimensionGenerator,
    WallerDimenGenerator,
    DimensionGeneratorBis,
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
        [
            [0.19215367, 0.07447659, 0.34677997, 0.61999938],
            [0.85769497, 0.47007512, 0.5157554, 0.82719861],
            [0.27367439, 0.63112969, 0.22969539, 0.36045082],
            [0.24980312, 0.63982306, 0.3957246, 0.83138228],
            [0.04923169, 0.79540832, 0.43595351, 0.75005156],
            [0.29421704, 0.60441345, 0.94073198, 0.0629855],
            [0.18569706, 0.10222294, 0.88576099, 0.85773229],
            [0.90366641, 0.60486249, 0.76656001, 0.88252291],
            [0.58484002, 0.52925899, 0.88195509, 0.06965376],
            [0.14643428, 0.62119783, 0.76216916, 0.95629683],
            [0.04106485, 0.99918406, 0.73537739, 0.81619452],
            [0.39920669, 0.74431558, 0.87749139, 0.39754929],
            [0.45358082, 0.01967785, 0.96484697, 0.91349083],
            [0.42911647, 0.41932921, 0.23221466, 0.81632824],
            [0.22812161, 0.81684781, 0.55631248, 0.59865408],
        ],
        [
            [0.28313848, 0.59268568, 0.05956726, 0.04822129],
            [0.95977662, 0.22590825, 0.29844012, 0.18701312],
            [0.65707627, 0.365417, 0.92249091, 0.47626297],
            [0.01293587, 0.39479071, 0.52114564, 0.63544407],
            [0.55178385, 0.54603529, 0.12101528, 0.4989738],
            [0.44461144, 0.03901759, 0.3739284, 0.95110758],
            [0.97532658, 0.35431873, 0.92876482, 0.17699203],
            [0.43302475, 0.21895238, 0.55413941, 0.73901917],
            [0.99943222, 0.45385158, 0.01548859, 0.32929804],
            [0.01967204, 0.37949068, 0.32320909, 0.49472875],
        ],
        [
            [0.54016259, 0.9969734, 0.78633924, 0.51086746],
            [0.52463354, 0.17930344, 0.36342232, 0.87286109],
            [0.14473518, 0.76086176, 0.97967552, 0.93224901],
            [0.98971782, 0.09134706, 0.81563796, 0.68615051],
            [0.20279794, 0.9630127, 0.94172444, 0.01967728],
            [0.71691824, 0.55070056, 0.38732873, 0.64592775],
            [0.4606539, 0.32649811, 0.78281818, 0.08946668],
            [0.43051606, 0.46488021, 0.11126845, 0.10819245],
            [0.25167579, 0.64029733, 0.2362831, 0.46761765],
            [0.74864642, 0.66236588, 0.86489551, 0.96543154],
            [0.22338833, 0.69056609, 0.7087024, 0.06459568],
        ],
        [
            [0.67159379, 0.24747655, 0.21190193, 0.30819393],
            [0.94543164, 0.38710749, 0.0194008, 0.03248972],
            [0.16153647, 0.85655823, 0.57527923, 0.57750053],
            [0.47853086, 0.53968923, 0.95085027, 0.19105121],
            [0.45706151, 0.03633782, 0.22456682, 0.71720246],
            [0.707517, 0.31426998, 0.62025153, 0.75268831],
            [0.15427333, 0.59636046, 0.44592157, 0.43069972],
            [0.67652712, 0.10460976, 0.67159493, 0.7530859],
            [0.31577139, 0.65780574, 0.80171352, 0.82898091],
        ],
        [[1, 0, 0], [1, 1, 0], [-1, 1, 0], [0, 1, 0]],
        [
            [0.81049989, 0.75800248, 0.46743956],
            [0.89431296, 0.98042562, 0.25508806],
            [0.93242096, 0.14936398, 0.13710553],
            [0.16343822, 0.35634931, 0.03839982],
            [0.9901016, 0.5043855, 0.53506935],
            [0.54624725, 0.24676143, 0.17712266],
            [0.28147921, 0.57103532, 0.43374979],
            [0.36183006, 0.92961152, 0.31774761],
            [0.88173001, 0.39684465, 0.20814363],
            [0.11136749, 0.32075147, 0.66003828],
            [0.16037099, 0.99442771, 0.01292324],
            [0.60195511, 0.83956336, 0.77152117],
            [0.60722621, 0.47713608, 0.16765088],
            [0.02960345, 0.42613379, 0.62137822],
            [0.94375555, 0.43622633, 0.5214306],
            [0.98922213, 0.06723254, 0.43452633],
            [0.89050629, 0.67509866, 0.61825686],
            [0.49098039, 0.05591642, 0.80899459],
            [0.29578949, 0.16328082, 0.25319735],
            [0.3678483, 0.80757164, 0.2959302],
            [0.77552794, 0.97924767, 0.37082694],
            [0.67173534, 0.58197531, 0.11081306],
            [0.18468571, 0.78687028, 0.491988],
            [0.65610139, 0.57034281, 0.33723567],
            [0.64978004, 0.603808, 0.36632189],
        ],
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
def dimen_name():
    return "ness_score"


@pytest.fixture()
def similarities(embeddings, seeds):
    generator = DimensionGenerator(embeddings, nn_n=10, k=10, chunk_size=10)
    directions = generator.nearest_neighbours_directions()
    seed_direction = generator.calculate_direction(seeds[0])
    return generator.nn_similarities(directions, seed_direction)


@pytest.fixture()
def similarities_bis(embeddings, seeds):
    generator = DimensionGeneratorBis(embeddings, nn_n=10, k=10, chunk_size=10)
    directions = generator.nearest_neighbours_directions()
    seed_direction = generator.calculate_direction(seeds[0])
    return generator.nn_similarities(directions, seed_direction)


@pytest.fixture()
def scores(embeddings, seeds, dimen_name):
    generator = DimensionGenerator(embeddings,seeds, [dimen_name], 
                                   nn_n=10, k=10, chunk_size=10)
    ranking = generator.value()[dimen_name]
    return ranking.sort_values()


@pytest.fixture()
def scores_bis(embeddings, seeds, dimen_name):
    generator = DimensionGeneratorBis(embeddings, nn_n=10, k=10, chunk_size=10)
    ranking = generator.get_scores_from_seeds(seeds, [dimen_name])[dimen_name]
    return ranking.sort_values()


@pytest.fixture()
def waller_scores(embeddings, seeds, dimen_name):
    dimen_generator = WallerDimenGenerator(embeddings)  # nn_n = 10, k = 10
    ranking = dimen_generator.get_scores(seeds, [dimen_name])[dimen_name]
    return ranking.sort_values()


class TestDimensionGenerator:
    def test_scores(self, scores, waller_scores):
        tm.assert_series_equal(scores, waller_scores)

    def test_score_in_range(self, scores):
        assert (scores <= 1).all()
        assert (scores >= -1).all()

    @pytest.mark.wip
    def test_similarity_in_range(self, similarities):
        assert (similarities <= 1).all()
        assert (similarities >= -1).all()


class TestDimensionGeneratorBis:
    @pytest.mark.wip
    def test_order(self, scores, scores_bis):
        assert (scores.index == scores_bis.index).all()

    @pytest.mark.wip
    def test_similarities(self, similarities, similarities_bis):
        assert similarities == similarities_bis
