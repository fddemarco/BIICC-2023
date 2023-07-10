import pathlib
import unittest
import pytest
from experiments.ranking import Ranking, calc_rbo
import scipy


def complete_ranking():
    return Ranking(
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


class RankingTestCase(unittest.TestCase):
    def test_kendall1(self):
        ranking = Ranking({'Conservative': 2, 'Christians': 1, 'GunsAreCool': -1, 'democrats': -2})
        self.assertAlmostEqual(1.0, ranking.kendall_score())

    def test_kendall2(self):
        ranking = Ranking({'Conservative': 2, 'Christians': -1, 'GunsAreCool': 1, 'democrats': -2})
        self.assertAlmostEqual(0.6666666666666669, ranking.kendall_score())

    def test_kendall3(self):
        ranking = Ranking({'Conservative': -3, 'Christians': 1, 'GunsAreCool': -1, 'democrats': -2})
        self.assertAlmostEqual(0.0, ranking.kendall_score())

    def test_kendall4(self):
        ranking = Ranking({'Conservative': -3, 'Christians': -1, 'GunsAreCool': 1, 'democrats': 2})
        self.assertAlmostEqual(-1.0, ranking.kendall_score())

    def test_kendall5(self):
        ranking = complete_ranking()
        self.assertAlmostEqual(0.9894736842105263, ranking.kendall_score())

    def test_calc_rbo_sanity_1(self):
        ranking = complete_ranking()
        self.assertAlmostEqual(
            1.0,
            calc_rbo(
                ranking.arxiv_waller_ranking(),
                ranking.arxiv_waller_ranking()
            )
        )

    def test_calc_rbo_sanity_2(self):
        ranking = complete_ranking()
        self.assertAlmostEqual(
            1.0,
            calc_rbo(
                ranking.arxiv_waller_ranking(),
                ranking.arxiv_waller_ranking()[:7]
            )
        )

    def test_calc_rbo_sanity_3(self):
        ranking = complete_ranking()
        predicted = ['fun 1', 'fun 2']
        self.assertAlmostEqual(
            0.0,
            calc_rbo(
                predicted,
                ranking.arxiv_waller_ranking()[:7]
            )
        )

    def test_calc_rbo_one_element_concordant(self):
        x = ['Conservative']
        y = ['Conservative']
        self.assertAlmostEqual(
            1.0,
            calc_rbo(
                x,
                y
            )
        )

    def test_calc_rbo_one_element_discordant(self):
        x = ['Conservative']
        y = ['democrats']
        self.assertAlmostEqual(
            0.0,
            calc_rbo(
                x,
                y
            )
        )

    def test_calc_rbo_1(self):
        x = ['democrats', 'EnoughLibertarianSpam', 'hillaryclinton', 'progressive', 'BlueMidterm2018']
        y = ['EnoughLibertarianSpam', 'democrats', 'hillaryclinton', 'progressive', 'BlueMidterm2018']
        self.assertAlmostEqual(
            0.8,
            calc_rbo(
                x,
                y
            )
        )

    def test_calc_rbo_2(self):
        y = ['democrats', 'EnoughLibertarianSpam', 'hillaryclinton', 'progressive', 'BlueMidterm2018']
        x = ['democrats', 'EnoughLibertarianSpam', 'hillaryclinton', 'BlueMidterm2018', 'progressive']
        self.assertAlmostEqual(
            0.95,
            calc_rbo(
                x,
                y
            )
        )

    def test_rbo_score_1(self):
        ranking = Ranking({
            'democrats': -3,
            'EnoughLibertarianSpam': -2,
            'hillaryclinton': -1,
            'BlueMidterm2018': 0,
            'progressive': 1
        })
        self.assertAlmostEqual(0.95, ranking.rbo_score())

    def test_rbo_score_2(self):
        ranking = Ranking({
            'EnoughLibertarianSpam': -3,
            'democrats': -2,
            'hillaryclinton': -1,
            'progressive': 0,
            'BlueMidterm2018': 1
        })
        self.assertAlmostEqual(0.8, ranking.rbo_score())
        print(scipy.__version__)

    def save_plot(self, plot_name):
        ranking = complete_ranking()
        plot = getattr(ranking, plot_name)()
        path = pathlib.Path('plots')
        path.mkdir(exist_ok=True)
        plot.savefig(
            path / f"{plot_name}_test.png",
            dpi=300,
            bbox_inches='tight')

    @pytest.mark.slow
    def test_violin_plot(self):
        self.save_plot('violin_plot')

    @pytest.mark.slow
    def test_bean_plot(self):
        self.save_plot('bean_plot')

    @pytest.mark.slow
    def test_kde_plot(self):
        self.save_plot('kde_plot')

    @pytest.mark.slow
    def test_auc_roc_plot(self):
        self.save_plot('roc_auc_plot')

    def test_auc_roc_score(self):
        ranking = complete_ranking()
        score = ranking.roc_auc_score()
        self.assertAlmostEqual(0.99, score)

    def test_t_student_p_value(self):
        ranking = complete_ranking()
        p_value = ranking.t_student_p_value()
        self.assertAlmostEqual(1.6477894923155955e-06, p_value)


if __name__ == '__main__':
    unittest.main()
