import statistics

import matplotlib.pyplot as plt
import pandas as pd
import rbo
from scipy import stats
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score


def leaning_right(z_score):
    return z_score > 1


def leaning_left(z_score):
    return z_score < -1


def calc_rbo(predicted_ranking, true_ranking):
    return rbo.RankingSimilarity(true_ranking, predicted_ranking).rbo()


def calc_mean(fst_half_score, snd_half_score):
    return statistics.mean([fst_half_score, snd_half_score])


def split_and_reverse(n, ranking):
    half_ranking = ranking[n:]
    half_ranking.reverse()
    return half_ranking


class Ranking:
    def __init__(self, ranking: dict):
        """
            :param ranking: Para cada subreddit, nos dice su score. {'Conservative': -0.228}
        """
        self.ranking = ranking

    # Classification metrics

    def z_score_for(self, subreddit):
        return (self.score_for(subreddit) - self.mean_score()) / self.sd_score()

    def score_for(self, subreddit):
        return self.ranking[subreddit]

    def scores(self):
        return self.ranking.values()

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
            average=None
        )
        return {
            label: scores[i]
            for i, label in enumerate(political_party_labels())
        }

    def precision_score(self):
        return self.classification_score(precision_score)

    def recall_score(self):
        return self.classification_score(recall_score)

    def f1_score(self):
        return self.classification_score(f1_score)

    def ground_truth(self):
        return [waller_political_party_label_for(subreddit) for subreddit in self.subreddits()]

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
        res = calc_rbo(predicted_ranking, true_ranking)
        return res

    def half_and_half_rbo_score(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = self.arxiv_waller_ranking()
        n = len(predicted_ranking) // 2

        fst_half_score = calc_rbo(predicted_ranking[:n], true_ranking[:n])
        snd_half_score = calc_rbo(split_and_reverse(n, predicted_ranking),
                                  split_and_reverse(n, true_ranking))
        return calc_mean(fst_half_score, snd_half_score)

    def two_way_rbo_score(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        true_ranking = self.arxiv_waller_ranking()
        desc_way_score = calc_rbo(predicted_ranking, true_ranking)

        predicted_ranking.reverse()
        true_ranking.reverse()
        asc_way_score = calc_rbo(predicted_ranking, true_ranking)
        return calc_mean(desc_way_score, asc_way_score)

    def subreddits_sorted_by_score_desc(self):
        return sorted(self.subreddits(), key=lambda k: self.score_for(k))

    def arxiv_waller_ranking(self):
        return arxiv_waller_ranking_for(self.subreddits())

    def bump_plot(self):
        predicted_ranking = self.subreddits_sorted_by_score_desc()
        waller_ranking = arxiv_waller_ranking_for(predicted_ranking)
        rankings = [
            {
                "Model": ["waller", "fasttext"],
                "Rank": [i + 1, predicted_ranking.index(subreddit) + 1],
                "Subreddit": subreddit
            } for i, subreddit in enumerate(waller_ranking)
        ]
        return bump_chart(rankings, len(waller_ranking))

    def violin_plot(self):
        df = pd.DataFrame(
            {'dem_rep': self.scores(),
             'subreddit': self.subreddits(),
             'political party': self.subreddits_party_labels()
             }
        )
        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            sns.violinplot(data=df, y='dem_rep', x='political party', inner='stick', ax=axis)
            return fig

    def bean_plot(self):
        df = pd.DataFrame(
            {'dem_rep': self.scores(),
             'subreddit': self.subreddits(),
             'political_party': self.subreddits_party_labels(),
             'political party': [''] * len(self.scores())
             }
        )
        with sns.plotting_context("paper"):
            fig, axis = plt.subplots()
            sns.violinplot(
                data=df,
                x='political party',
                y='dem_rep',
                hue='political_party',
                split=True,
                inner='stick',
                ax=axis)
            return fig

    def subreddits_party_labels(self):
        return [waller_political_party_label_for(s) for s in self.subreddits()]


def democrat_label():
    return 'democrat'


def conservative_label():
    return 'conservative'


def political_party_labels():
    return [democrat_label(), neutral_label(), conservative_label()]


def neutral_label():
    return 'neutral'


def waller_political_party_label_for(subreddit):
    labels = arxiv_waller_labels()
    return labels[subreddit]


def arxiv_waller_labels():
    return {
        'democrats': democrat_label(),
        'EnoughLibertarianSpam': democrat_label(),
        'hillaryclinton': democrat_label(),
        'progressive': democrat_label(),
        'BlueMidterm2018': democrat_label(),
        'EnoughHillHate': democrat_label(),
        'Enough_Sanders_Spam': democrat_label(),
        'badwomensanatomy': democrat_label(),
        'racism': democrat_label(),
        'GunsAreCool': democrat_label(),
        'Christians': conservative_label(),
        'The_Farage': conservative_label(),
        'new_right': conservative_label(),
        'conservatives': conservative_label(),
        'metacanada': conservative_label(),
        'Mr_Trump': conservative_label(),
        'NoFapChristians': conservative_label(),
        'TrueChristian': conservative_label(),
        'The_Donald': conservative_label(),
        'Conservative': conservative_label()
    }


def arxiv_waller_ranking():
    return arxiv_waller_labels().keys()


def arxiv_waller_ranking_for(subreddits):
    return [s for s in arxiv_waller_ranking()
            if s in subreddits]


def bump_chart(elements, n):
    fig, ax = plt.subplots(figsize=(12, 6))
    for element in elements:
        ax.plot(
            element["Model"],
            element["Rank"],
            "o-",
            markerfacecolor="white",
            linewidth=3
        )
        ax.annotate(
            element["Subreddit"],
            xy=("fasttext", element["Rank"][1]),
            xytext=(1.01, element["Rank"][1])
        )
        ax.annotate(
            element["Subreddit"],
            xy=("waller", element["Rank"][0]),
            xytext=(-0.3, element["Rank"][0])
        )

    plt.gca().invert_yaxis()
    plt.yticks([i for i in range(1, n + 1)])

    ax.set_xlabel('Model')
    ax.set_ylabel('Rank')
    ax.set_title('Comparison of Models on Subreddit Classification Task')

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return plt
