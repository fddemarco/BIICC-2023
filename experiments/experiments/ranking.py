import statistics
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rbo
from scipy import stats
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
    ndcg_score,
)
from sklearn.preprocessing import MinMaxScaler


def leaning_right(z_score):
    return z_score > 1


def leaning_left(z_score):
    return z_score < -1


def calc_rbo(predicted_ranking, true_ranking, p):  # TODO: Seleccionar un p
    return rbo.RankingSimilarity(true_ranking, predicted_ranking).rbo(p=p)


def calc_mean(fst_half_score, snd_half_score):
    return statistics.mean([fst_half_score, snd_half_score])


def split_and_reverse(n, ranking):
    half_ranking = ranking[n:]
    half_ranking.reverse()
    return half_ranking


def dem_rep_field():
    return "dem_rep"


class Ranking:
    @classmethod
    def from_pandas(cls, score_data, p=1.0):
        score_data = score_data[score_data.index.isin(arxiv_waller_ranking())].copy()
        score_data = score_data.to_dict(orient="dict")[dem_rep_field()]
        return cls(score_data, p)

    def __init__(self, ranking: dict, p:int = 1.0):
        """
        :param ranking: Para cada subreddit, nos dice su score. {'Conservative': -0.228}
        """
        self.ranking = ranking
        self.p = p

    def compare_ranking(self):
        classification_metrics = self.evaluate_classification_metrics()
        ranking_metrics = self.evaluate_ranking_metrics()
        plots = self.generate_plots()

        Metrics = namedtuple("Metrics", "classification_metrics ranking_metrics plots")
        return Metrics(classification_metrics, ranking_metrics, plots)

    def generate_plots(self):
        return {
            "bump": self.bump_plot(),
            "kde": self.kde_plot(),
            "violin": self.violin_plot(),
            "bean": self.bean_plot(),
            "roc_auc": self.roc_auc_plot(),
        }

    def evaluate_ranking_metrics(self):
        return {
            "Kendall Tau": [self.kendall_score()],
            "Classic RBO": [self.rbo_score()],
            "Two way RBO": [self.two_way_rbo_score()],
            "H&H RBO": [self.half_and_half_rbo_score()],
            "AUC ROC": [self.roc_auc_score()],
            "nDCG": [self.n_dcg_score()],
        }

    def evaluate_classification_metrics(self):
        return {
            "Precision": self.precision_score(),
            "Recall": self.recall_score(),
            "F1 Score": self.f1_score(),
        }

    # Classification metrics

    def z_score_for(self, subreddit):
        return (self.score_for(subreddit) - self.mean_score()) / self.sd_score()

    def z_scores(self):
        return self.z_scores_for_subreddits(self.subreddits())

    def z_scores_for_subreddits(self, subreddits):
        return [self.z_score_for(s) for s in subreddits]

    def score_for(self, subreddit):
        return self.ranking[subreddit]

    def scores(self):
        return self.scores_for_subreddits(self.subreddits())

    def scores_for_subreddits(self, subreddits):
        return [self.score_for(s) for s in subreddits]

    def party_scores(self, party_label):
        partisans = self.subreddits_of_partisan(party_label)
        return np.array(self.z_scores_for_subreddits(partisans))

    def democrats_scores(self):
        return self.party_scores(democrat_label())

    def conservatives_scores(self):
        return self.party_scores(conservative_label())

    def subreddits(self):
        return self.ranking.keys()

    def mean_score(self):
        return statistics.mean(self.scores())

    def sd_score(self):
        return statistics.stdev(self.scores())

    def classification_score(self, scoring_function):
        scores = scoring_function(
            self.ground_truth(),
            self.predict_political_party(),
            labels=political_party_labels(),
            average=None,
        )
        return {label: scores[i] for i, label in enumerate(political_party_labels())}

    def precision_score(self):
        return self.classification_score(precision_score)

    def recall_score(self):
        return self.classification_score(recall_score)

    def f1_score(self):
        return self.classification_score(f1_score)

    def ground_truth(self):
        return [
            waller_political_party_label_for(subreddit)
            for subreddit in self.subreddits()
        ]

    def predict_political_party(self):
        predicted_labels = []
        for subreddit in self.subreddits():
            z_score = self.z_score_for(subreddit)
            if leaning_right(z_score):
                predicted_labels.append(conservative_label())
            elif leaning_left(z_score):
                predicted_labels.append(democrat_label())
            else:
                predicted_labels.append(neutral_label())

        return predicted_labels

    # Ranking comparison metrics

    def kendall_score(self):
        """
        -1 = negatively correlated,
         0 = random,
         1 = positively correlated
        """
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = arxiv_waller_ranking_for(predicted_ranking)
        x = [true_ranking.index(item) + 1 for item in predicted_ranking]
        y = [i for i in range(1, len(true_ranking) + 1)]
        res = stats.kendalltau(x, y)
        return res.statistic

    def rbo_score(self):
        """
        0 = disjoint,
        1 = identical
        """
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = self.arxiv_waller_ranking()
        res = calc_rbo(predicted_ranking, true_ranking, self.p)
        return res

    def half_and_half_rbo_score(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = self.arxiv_waller_ranking()
        n = len(predicted_ranking) // 2

        fst_half_score = calc_rbo(predicted_ranking[:n], true_ranking[:n], self.p)
        snd_half_score = calc_rbo(
            split_and_reverse(n, predicted_ranking),
            split_and_reverse(n, true_ranking),
            self.p,
        )
        return calc_mean(fst_half_score, snd_half_score)

    def two_way_rbo_score(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = self.arxiv_waller_ranking()
        desc_way_score = calc_rbo(predicted_ranking, true_ranking, self.p)

        predicted_ranking.reverse()
        true_ranking.reverse()
        asc_way_score = calc_rbo(predicted_ranking, true_ranking, self.p)
        return calc_mean(desc_way_score, asc_way_score)

    def t_student_p_value(self):
        p_val_t = stats.ttest_ind(self.conservatives_scores(), self.democrats_scores())[
            1
        ]
        return p_val_t

    def roc_auc_score(self):
        standardized_scores = np.array(self.z_scores())
        probabilities = 1 / (1 + np.exp(-standardized_scores))
        labels = [
            1 if label == conservative_label() else 0
            for label in self.subreddits_party_labels()
        ]
        roc_auc = roc_auc_score(labels, probabilities)
        return roc_auc

    def n_dcg_score(self):
        data = self.score_data()
        scaled_data = self.min_max_scale(data)
        return ndcg_score([scaled_data.waller_scores], [scaled_data.scores])

    def score_data(self):
        waller_scores = [arxiv_waller_score_for(s) for s in self.subreddits()]
        data = pd.DataFrame({"scores": self.scores(), "waller_scores": waller_scores})
        return data

    def min_max_scale(self, data):
        scaler = MinMaxScaler()
        scaler.fit(data)
        scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
        return scaled_data

    def subreddits_sorted_by_score_desc(self):
        return sorted(self.subreddits(), key=lambda k: self.score_for(k))

    def arxiv_waller_ranking(self):
        return arxiv_waller_ranking_for(self.subreddits())

    # Plots

    def bump_plot(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        waller_ranking = arxiv_waller_ranking_for(predicted_ranking)
        rankings = [
            {
                "Model": ["waller", "fasttext"],
                "Rank": [i + 1, predicted_ranking.index(subreddit) + 1],
                "Subreddit": subreddit,
            }
            for i, subreddit in enumerate(waller_ranking)
        ]
        with sns.plotting_context("paper"):
            return bump_chart(rankings, len(waller_ranking))

    def violin_plot(self, title=None):
        df = pd.DataFrame(
            {
                "dem_rep": self.scores(),
                "subreddit": self.subreddits(),
                "political party": self.subreddits_party_labels(),
            }
        )
        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            sns.violinplot(
                data=df,
                y="dem_rep",
                x="political party",
                order=[democrat_label(), conservative_label()],
                inner="stick",
                ax=axis,
            )
            axis.set_title(title)
            return fig

    def kde_plot(self):
        df = pd.DataFrame(
            {
                "dem_rep": self.scores(),
                "subreddit": self.subreddits(),
                "political party": self.subreddits_party_labels(),
            }
        )
        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            sns.kdeplot(data=df, x="dem_rep", hue="political party", fill=True, ax=axis)
            return fig

    def bean_plot(self):
        df = pd.DataFrame(
            {
                "dem_rep": self.scores(),
                "subreddit": self.subreddits(),
                "political_party": self.subreddits_party_labels(),
                "political party": [""] * len(self.scores()),
            }
        )
        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            sns.violinplot(
                data=df,
                x="political party",
                y="dem_rep",
                hue="political_party",
                split=True,
                inner="stick",
                ax=axis,
            )
            return fig

    def roc_auc_plot(self):
        # Genera dos distribuciones, una para la "clase" positiva y otra para la negativa
        negative = self.democrats_scores()
        positive = self.conservatives_scores()

        # Junta los datos en un solo arreglo y crea las etiquetas correspondientes
        data = np.concatenate([positive, negative])
        labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])

        # Calcula la curva ROC
        fpr, tpr, _ = roc_curve(labels, data)
        roc_auc = auc(fpr, tpr)

        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            axis.plot(
                fpr, tpr, color="darkorange", label="Curva ROC (area = %0.2f)" % roc_auc
            )
            axis.plot(
                [0, 1], [0, 1], color="navy", linestyle="--"
            )  # línea diagonal para comparación
            axis.set_xlim([0.0, 1.0])
            axis.set_ylim([0.0, 1.05])
            axis.set_xlabel("Tasa de Falsos Positivos")
            axis.set_ylabel("Tasa de Verdaderos Positivos")
            axis.set_title("Curva ROC de las distribuciones")
            axis.legend(loc="lower right")
            return fig

    def subreddits_party_labels(self):
        return [waller_political_party_label_for(s) for s in self.subreddits()]

    def subreddits_of_partisan(self, party_label):
        return [
            s
            for s in self.subreddits()
            if waller_political_party_label_for(s) == party_label
        ]


def democrat_label():
    return "democrat"


def conservative_label():
    return "conservative"


def political_party_labels():
    return [democrat_label(), neutral_label(), conservative_label()]


def neutral_label():
    return "neutral"


def waller_political_party_label_for(subreddit):
    score = arxiv_waller_scores()[subreddit]
    if score < 0:
        return democrat_label()
    return conservative_label()


def arxiv_waller_labels():
    return {
        subreddit: waller_political_party_label_for(subreddit)
        for subreddit in arxiv_waller_ranking()
    }


def arxiv_waller_scores():
    return {
        "democrats": -0.345948606707049,
        "EnoughLibertarianSpam": -0.322594981636269,
        "hillaryclinton": -0.3027931218773805,
        "progressive": -0.2994712557588187,
        "BlueMidterm2018": -0.2977831668625458,
        "EnoughHillHate": -0.2933539740564371,
        "Enough_Sanders_Spam": -0.2929483022563205,
        "badwomensanatomy": -0.2926874460908718,
        "racism": -0.2921137058022828,
        "GunsAreCool": -0.290219904193626,
        "Christians": 0.2607635855569176,
        "The_Farage": 0.2658256024989052,
        "new_right": 0.2697649330292293,
        "conservatives": 0.2743712713632447,
        "metacanada": 0.2865165930755363,
        "Mr_Trump": 0.2895610652703748,
        "NoFapChristians": 0.2934370114397415,
        "TrueChristian": 0.3142461533194396,
        "The_Donald": 0.3351316374970578,
        "Conservative": 0.444171415963574,
    }


def arxiv_waller_score_for(subreddit):
    return arxiv_waller_scores()[subreddit]


def arxiv_waller_ranking():
    return list(arxiv_waller_scores().keys())


def arxiv_waller_ranking_for(subreddits):
    return [s for s in arxiv_waller_ranking() if s in subreddits]


def bump_chart(elements, n):
    fig, ax = plt.subplots()
    for element in elements:
        ax.plot(
            element["Model"],
            element["Rank"],
            "o-",
            markerfacecolor="white",
            linewidth=3,
        )
        ax.annotate(
            element["Subreddit"],
            xy=("predicted", element["Rank"][1]),
            xytext=(1.01, element["Rank"][1]),
        )
        ax.annotate(
            element["Subreddit"],
            xy=("waller", element["Rank"][0]),
            xytext=(-0.3, element["Rank"][0]),
        )

    plt.gca().invert_yaxis()
    plt.yticks([i for i in range(1, n + 1)])

    ax.set_xlabel("Model")
    ax.set_ylabel("Rank")
    ax.set_title("Comparison of Models")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig
