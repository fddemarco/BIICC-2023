import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import rbo
from scipy import stats
import statistics


class Ranking:
    def __init__(self, ranking: dict):
        """
            :param ranking: Para cada subreddit, nos dice su score. {'Conservative': -0.228}
        """
        self.ranking = ranking

    def z_score(self, subreddit):
        return (self.score_for(subreddit) - self.mean_score()) / self.sd_score()

    def score_for(self, subreddit):
        return self.ranking[subreddit]

    def scores(self):
        return self.ranking.values()

    def mean_score(self):
        return statistics.mean(self.ranking.values())

    def sd_score(self):
        return statistics.stdev(self.ranking.values())

    def classification_score(self, scoring_function):
        scores = scoring_function(
                    self.ground_truth(),
                    self.predict_partisan(),
                    labels=partisan_labels(),
                    average=None
        )
        return {
            label: scores[i]
            for i, label in enumerate(partisan_labels())
        }

    def precision_score(self):
        return self.classification_score(precision_score)

    def recall_score(self):
        return self.classification_score(recall_score)

    def f1_score(self):
        return self.classification_score(f1_score)

    # average precision no soporta multi-class

    def ground_truth(self):
        return [waller_label_for(subreddit) for subreddit in self.ranked_subreddits()]

    def predict_partisan(self):
        predicted_labels = []
        for subreddit in self.ranked_subreddits():
            z_score = self.z_score(subreddit)
            if self.leaning_right(z_score):
                predicted_labels.append(conservative_label())
            elif self.leaning_left(z_score):
                predicted_labels.append(democrat_label())
            else:
                predicted_labels.append(neutral_label())

        return predicted_labels

    def leaning_right(self, z_score):
        return z_score > 1

    def leaning_left(self, z_score):
        return z_score < -1

    def kendall_score(self):
        # -1 = neg correlated, 0 = random, 1 = pos correlated
        predicted_ranking = self.predict_partisan_ranking()
        true_ranking = arxiv_waller_ranking_for(predicted_ranking)
        res = stats.kendalltau(true_ranking, predicted_ranking)
        return res.statistic

    def rbo_score(self):
        # 0 = disjoint, 1 = identical
        predicted_ranking = self.predict_partisan_ranking()
        true_ranking = self.arxiv_waller_ranking()
        res = rbo.RankingSimilarity(true_ranking, predicted_ranking).rbo()
        return res

    def predict_partisan_ranking(self):
        return sorted(self.ranked_subreddits(), key=lambda k: self.score_for(k))

    def ranked_subreddits(self):
        return self.ranking.keys()

    def normalized_dcg_score(self):
        predicted_ranking = self.predict_partisan_ranking()
        true_ranking = self.arxiv_waller_ranking()

        predicted_relevance = self.normalized_dcg_rank_relevance(predicted_ranking)
        true_relevance = self.normalized_dcg_rank_relevance(true_ranking)

        return ndcg_score(np.asarray([true_relevance]), np.asarray([predicted_relevance]))

    def normalized_dcg_rank_relevance(self, predicted_ranking):
        return [self.ndcg_relevance_for(predicted_ranking, s) for s in self.ranked_subreddits()]

    def ndcg_relevance_for(self, predicted_ranking, s):
        n = len(predicted_ranking)
        return n - predicted_ranking.index(s)

    def arxiv_waller_ranking(self):
        return arxiv_waller_ranking_for(self.ranked_subreddits())


def democrat_label():
    return 'democrat'


def conservative_label():
    return 'conservative'


def partisan_labels():
    return [democrat_label(), neutral_label(), conservative_label()]


def neutral_label():
    return 'neutral'


def waller_label_for(subreddit):
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
