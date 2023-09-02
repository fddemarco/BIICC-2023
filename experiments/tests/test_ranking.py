import pathlib

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
    return pd.DataFrame(data, index=["democrats", "Conservative"])


class TestRanking():
    @pytest.mark.parametrize(
        "ranking_scores, expected_tau",
        [({"Conservative": 2, "Christians": 1, "GunsAreCool": -1, "democrats": -2}, 1.0),
        ({"Conservative": 2, "Christians": -1, "GunsAreCool": 1, "democrats": -2}, .6666666666666669),
        ({"Conservative": -3, "Christians": 1, "GunsAreCool": -1, "democrats": -2}, .0),
        ({"Conservative": -3, "Christians": -1, "GunsAreCool": 1, "democrats": 2}, -1.0),
        (arxiv_waller_scores(), 1.0)
        ]
    )
    def test_kendall_tau(self, ranking_scores, expected_tau):
        ranking = Ranking(ranking_scores)
        assert expected_tau == pytest.approx(ranking.kendall_score())

    def test_calc_rbo_sanity_same_ranking(self):
        ranking = waller_ranking()
        assert 1.0 == pytest.approx(
            calc_rbo(ranking.arxiv_waller_ranking(), ranking.arxiv_waller_ranking()),
        )

    def test_calc_rbo_sanity_sub_chain(self):
        ranking = waller_ranking()
        assert 1.0 == pytest.approx(
            calc_rbo(
                ranking.arxiv_waller_ranking(), ranking.arxiv_waller_ranking()[:7]
            )
        )

    def test_calc_rbo_sanity_disjoint(self):
        ranking = waller_ranking()
        predicted = ["fun 1", "fun 2"]
        assert 0.0 == pytest.approx(calc_rbo(predicted, ranking.arxiv_waller_ranking()))

    def test_calc_rbo_one_element_concordant(self):
        x = ["Conservative"]
        y = ["Conservative"]
        assert 1.0 == pytest.approx(calc_rbo(x, y))

    def test_calc_rbo_one_element_discordant(self):
        x = ["Conservative"]
        y = ["democrats"]
        assert 0.0 == pytest.approx(calc_rbo(x, y))

    def test_calc_rbo_1(self):
        x = [
            "democrats",
            "EnoughLibertarianSpam",
            "hillaryclinton",
            "progressive",
            "BlueMidterm2018",
        ]
        y = [
            "EnoughLibertarianSpam",
            "democrats",
            "hillaryclinton",
            "progressive",
            "BlueMidterm2018",
        ]
        assert 0.8 == pytest.approx(calc_rbo(x, y))

    def test_calc_rbo_2(self):
        y = [
            "democrats",
            "EnoughLibertarianSpam",
            "hillaryclinton",
            "progressive",
            "BlueMidterm2018",
        ]
        x = [
            "democrats",
            "EnoughLibertarianSpam",
            "hillaryclinton",
            "BlueMidterm2018",
            "progressive",
        ]
        assert 0.95 == pytest.approx(calc_rbo(x, y))

    def test_rbo_score_1(self):
        ranking = Ranking(
            {
                "democrats": -3,
                "EnoughLibertarianSpam": -2,
                "hillaryclinton": -1,
                "BlueMidterm2018": 0,
                "progressive": 1,
            }
        )
        assert 0.95 == pytest.approx(ranking.rbo_score())

    def test_rbo_score_2(self):
        ranking = Ranking(
            {
                "EnoughLibertarianSpam": -3,
                "democrats": -2,
                "hillaryclinton": -1,
                "progressive": 0,
                "BlueMidterm2018": 1,
            }
        )
        assert 0.8 == pytest.approx(ranking.rbo_score())

    def save_plot(self, plot_name):
        ranking = waller_ranking()
        plot = getattr(ranking, plot_name)()
        path = pathlib.Path("plots")
        path.mkdir(exist_ok=True)
        plot.savefig(path / f"{plot_name}_test.png", dpi=300, bbox_inches="tight")

    @pytest.mark.slow
    def test_violin_plot(self):
        self.save_plot("violin_plot")

    @pytest.mark.slow
    def test_bean_plot(self):
        self.save_plot("bean_plot")

    @pytest.mark.slow
    def test_kde_plot(self):
        self.save_plot("kde_plot")

    @pytest.mark.slow
    def test_auc_roc_plot(self):
        self.save_plot("roc_auc_plot")

    def test_auc_roc_score(self):
        ranking = waller_ranking()
        score = ranking.roc_auc_score()
        assert 1.0 == pytest.approx(score)

    def test_t_student_p_value(self):
        ranking = waller_ranking()
        p_value = ranking.t_student_p_value()
        assert 0.0 == pytest.approx(p_value)

    def test_from_pandas(self):
        ranking = Ranking.from_pandas(ranking_as_dataframe({"dem_rep": [-2, 2]}))
        score = ranking.roc_auc_score()
        assert 1.0 == pytest.approx(score)

    def test_n_dcg_score_01(self):
        ranking = waller_ranking()
        assert 1.0 == pytest.approx(ranking.n_dcg_score())

    def test_n_dcg_sanity(self, monkeypatch):
        def mock_arxiv_waller_scores(*args, **kwargs):
            return {
                "s1": 10,
                "s2": 0,
                "s3": 0,
                "s4": 1,
                "s5": 5
                }

        monkeypatch.setattr("experiments.ranking.arxiv_waller_scores", mock_arxiv_waller_scores)
        ranking = Ranking({"s1": 0.1, "s2": 0.2, "s3": 0.3, "s4": 4, "s5": 70})
        assert 0.6956940443813076 == pytest.approx(ranking.n_dcg_score())


if __name__ == "__main__":
    unittest.main()
