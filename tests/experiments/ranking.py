import unittest
from src.experiments.ranking import Ranking


class RankingTestCase(unittest.TestCase):
    def test_kendall1(self):
        ranking = Ranking({"Conservative": 2, "Christians": 1, "GunsAreCool": -1, "democrats": -2})
        self.assertAlmostEquals(1.0, ranking.kendall_score())

    def test_kendall2(self):
        ranking = Ranking({"Conservative": 2, "Christians": -1, "GunsAreCool": 1, "democrats": -2})
        self.assertAlmostEquals(0.6666666666666669, ranking.kendall_score())

    def test_kendall3(self):
        ranking = Ranking({"Conservative": -3, "Christians": 1, "GunsAreCool": -1, "democrats": -2})
        self.assertAlmostEquals(0.0, ranking.kendall_score())

    def test_kendall4(self):
        ranking = Ranking({"Conservative": -3, "Christians": -1, "GunsAreCool": 1, "democrats": 2})
        self.assertAlmostEquals(-1.0, ranking.kendall_score())

    def test_kendall5(self):
        ranking = Ranking(
            {
                'democrats': -10,
                'EnoughLibertarianSpam': -9,
                'hillaryclinton': -8,
                'progressive': -7,
                'BlueMidterm2018': -6,
                'EnoughHillHate': -5,
                'Enough_Sanders_Spam': -4,
                'badwomensanatomy': -3,
                'racism': -2,
                'GunsAreCool': 1,
                'Christians': -1,
                'The_Farage': 2,
                'new_right': 3,
                'conservatives': 4,
                'metacanada': 5,
                'Mr_Trump': 6,
                'NoFapChristians': 7,
                'TrueChristian': 8,
                'The_Donald': 9,
                'Conservative': 10
            })
        self.assertAlmostEquals(0.9894736842105263, ranking.kendall_score())


if __name__ == '__main__':
    unittest.main()
