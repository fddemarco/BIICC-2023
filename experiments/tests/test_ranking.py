import pathlib
import unittest
from unittest.mock import patch

import pandas as pd
import pytest
from experiments.ranking import Ranking, calc_rbo, arxiv_waller_scores


def waller_ranking():
    return Ranking(arxiv_waller_scores())


def reverse_waller_ranking():
    subreddits = list(arxiv_waller_scores().keys())
    scores = list(arxiv_waller_scores().values())
    scores.reverse()
    data = {s: scores[i] for i, s in enumerate(subreddits)}
    return Ranking(data)


def ranking_as_dataframe(data):
    return pd.DataFrame(data, index=['democrats', 'Conservative'])


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
        ranking = waller_ranking()
        self.assertAlmostEqual(1.0, ranking.kendall_score())

    def test_calc_rbo_sanity_same_ranking(self):
        ranking = waller_ranking()
        self.assertAlmostEqual(
            1.0,
            calc_rbo(
                ranking.arxiv_waller_ranking(),
                ranking.arxiv_waller_ranking()
            )
        )

    def test_calc_rbo_sanity_sub_chain(self):
        ranking = waller_ranking()
        self.assertAlmostEqual(
            1.0,
            calc_rbo(
                ranking.arxiv_waller_ranking(),
                ranking.arxiv_waller_ranking()[:7]
            )
        )

    def test_calc_rbo_sanity_disjoint(self):
        ranking = waller_ranking()
        predicted = ['fun 1', 'fun 2']
        self.assertAlmostEqual(
            0.0,
            calc_rbo(
                predicted,
                ranking.arxiv_waller_ranking()
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

    def save_plot(self, plot_name):
        ranking = waller_ranking()
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
        ranking = waller_ranking()
        score = ranking.roc_auc_score()
        self.assertAlmostEqual(1.0, score)

    def test_t_student_p_value(self):
        ranking = waller_ranking()
        p_value = ranking.t_student_p_value()
        self.assertAlmostEqual(0.0, p_value)

    def test_from_pandas(self):
        ranking = Ranking.from_pandas(ranking_as_dataframe({'dem_rep': [-2, 2]}))
        score = ranking.roc_auc_score()
        self.assertAlmostEqual(1.0, score)

    def test_n_dcg_score_01(self):
        ranking = waller_ranking()
        self.assertAlmostEqual(1.0, ranking.n_dcg_score())

    def test_n_dcg_sanity(self):
        with patch('experiments.ranking.arxiv_waller_scores') as mocked_function:
            mocked_function.return_value = {'s1': 10, 's2': 0, 's3': 0, 's4': 1, 's5': 5}
            ranking = Ranking({'s1': .1, 's2': .2, 's3': .3, 's4': 4, 's5': 70})
            self.assertAlmostEqual(0.6956940443813076, ranking.n_dcg_score())


if __name__ == '__main__':
    unittest.main()
