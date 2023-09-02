import pathlib

import pandas as pd
import pytest
from experiments.ranking import Ranking, calc_rbo, arxiv_waller_scores, arxiv_waller_ranking

@pytest.fixture()
def waller_ranking():
    return Ranking(arxiv_waller_scores())

@pytest.fixture()
def ranking_as_dataframe():
    return pd.DataFrame({"dem_rep": [-2, 2]}, index=["democrats", "Conservative"])


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

    @pytest.mark.parametrize(
        "ranking, ranking_other, expected_rbo",[
            (arxiv_waller_ranking(), arxiv_waller_ranking(), 1.0),
            (arxiv_waller_ranking(), ["disjoint 1", "disjoint 2"], .0),
            ([
                "democrats",
                "EnoughLibertarianSpam",
                "hillaryclinton",
                "progressive",
                "BlueMidterm2018",
            ], [
                "EnoughLibertarianSpam",
                "democrats",
                "hillaryclinton",
                "progressive",
                "BlueMidterm2018",
            ], .8
            )
        ]
    )
    def test_calc_rbo(self, ranking, ranking_other, expected_rbo):
        assert expected_rbo == pytest.approx(
            calc_rbo(ranking, ranking_other, 1.0),
        )

    @pytest.mark.parametrize(
        "scores, expected_rbo",[
            ({
                "democrats": -3,
                "EnoughLibertarianSpam": -2,
                "hillaryclinton": -1,
                "BlueMidterm2018": 0,
                "progressive": 1,
                }, .95
            ), 
            ({
                "EnoughLibertarianSpam": -3,
                "democrats": -2,
                "hillaryclinton": -1,
                "progressive": 0,
                "BlueMidterm2018": 1,
            }, .8)
        ]
    )
    def test_rbo_score(self, scores, expected_rbo):
        ranking = Ranking(scores)
        assert expected_rbo == pytest.approx(ranking.rbo_score(1.0))

    
    @pytest.mark.slow
    @pytest.mark.parametrize("plot_function, plot_name",
    [
        (lambda ranking: ranking.violin_plot(), "violin_plot"),
        (lambda ranking: ranking.bean_plot(), "bean_plot"),
        (lambda ranking: ranking.kde_plot(), "kde_plot"),
        (lambda ranking: ranking.roc_auc_plot(), "roc_auc_plot")
    ])
    def test_plots(self, waller_ranking, plot_function, plot_name):
        plot = plot_function(waller_ranking)
        path = pathlib.Path("plots")
        path.mkdir(exist_ok=True)
        plot.savefig(path / f"{plot_name}_test.png", dpi=300, bbox_inches="tight")

    def test_auc_roc_score(self, waller_ranking):
        score = waller_ranking.roc_auc_score()
        assert 1.0 == pytest.approx(score)

    def test_t_student_p_value(self, waller_ranking):
        p_value = waller_ranking.t_student_p_value()
        assert 0.0 == pytest.approx(p_value)

    def test_from_pandas(self, ranking_as_dataframe):
        ranking = Ranking.from_pandas(ranking_as_dataframe)
        score = ranking.roc_auc_score()
        assert 1.0 == pytest.approx(score)

    def test_n_dcg_score(self, waller_ranking):
        assert 1.0 == pytest.approx(waller_ranking.n_dcg_score())

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
