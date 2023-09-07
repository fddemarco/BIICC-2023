"""
    Experiment module
"""

import pathlib

import matplotlib.pyplot as plt
import pyarrow.dataset as ds
import pandas as pd

import fasttext
import experiments.dimension_generator as dg
import experiments.reddit_posts as rps
import experiments.ranking as rk


class Experiment:
    """
    Experiments Class
    """

    def __init__(self, year, working_dir, results_dir, post_type, dataset):
        self.year = str(year)
        self.working_dir = pathlib.Path(working_dir)
        self.results_folder = str(results_dir)
        self.post_type = post_type
        self.dataset = str(dataset)

    @property
    def _base_dataset_dir(self):
        return self._base_parent_dir / self.dataset

    @property
    def _base_parent_dir(self):
        return self.working_dir / self.posts_type / self.year

    @property
    def _data_pathname(self):
        return self._base_dataset_dir / "data"

    @property
    def _truncated_data_pathname(self):
        truncated_dir = self._base_parent_dir / "truncated"
        truncated_dir.mkdir(parents=True, exist_ok=True)
        return truncated_dir

    @property
    def _results_dir(self):
        results_dir = self._base_dataset_dir / self.results_folder
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def _result_path(self, filename):
        return self._results_dir / filename

    @property
    def _subreddits_pathname(self):
        return self._result_path("subreddits.txt")

    @property
    def _fasttext_model_pathname(self):
        return self._result_path("subreddits.bin")

    @property
    def _embedding_pathname(self):
        return self._result_path("embeddings.csv")

    def apply_texts(self):
        """
        Applies text command

        Generates a single file with all subreddits posts text.
        Each paragraph corresponds to a single subreddit.
        Only top 10k most relevant subreddits are taken in consideration.
        """
        reddit_posts = self._get_reddit_posts()
        reddit_posts.generate_text()

    def apply_truncate(self):
        """
        Applies truncate command

        Generates a truncated dataset by keeping only the most relevant posts of each subreddit.
        Each subreddit has a threshold of 10k bytes.
        Only top 10k most relevant subreddits are taken in consideration.
        """
        reddit_posts = self._get_reddit_posts()
        data = reddit_posts.truncate_dataset()
        data.to_parquet(self._truncated_data_pathname / "truncated_data.parquet")

    def _get_reddit_posts(self):
        dataset = ds.dataset(self._data_pathname, format="parquet")
        reddit_posts = rps.RedditPosts(dataset, self, self.post_type)
        return reddit_posts

    def write_text(self, text):
        """
        Consumes text of subreddit posts and appends it to a single file.
        """
        with open(self._subreddits_pathname, "a", encoding="utf-8") as file:
            file.write(text + "\n")

    def apply_embeddings(self):
        """
        Applies embeddings command

        Generates embeddings for each subreddit in the given dataset and saves them to disk.
        """
        model_pathname = str(self._fasttext_model_pathname.absolute())
        model = fasttext.load_model(model_pathname)
        reddit_posts = self._get_reddit_posts()

        embeddings = reddit_posts.generate_embeddings_for(model)
        embeddings.to_csv(self._embedding_pathname)

    def get_scores(self, data, seeds=None) -> pd.DataFrame:
        """
        Generates scores from DATA using SEEDS
        """
        if seeds is None:  # TODO: parametrizar names
            seeds = [("democrats", "Conservative")]
        generator = dg.DimensionGenerator(data)
        return generator.get_scores_from_seeds([self._dem_rep_field], seeds)

    def apply_compare(self):
        """
        Applies compare command

        Generates metrics and plots comparing our results to Waller's et al.
        Metrics include: RBO, Kendall Tau, T-Standard p-value, ROC AUC
        Plots include: bump chart, ROC AUC, violin plot, kde plot
        """
        data = pd.read_csv(self._embedding_pathname, index_col=0)
        ranking = rk.Ranking.from_pandas(self.get_scores(data))
        metrics = ranking.compare_ranking()

        self._save_metrics(metrics.classification_metrics, "classification_metrics.csv")
        self._save_metrics(metrics.ranking_metrics, "ranking_metrics.csv")

        for name, fig in metrics.plots.items():
            fig.savefig(self._result_path(f"{name}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    def _save_metrics(self, metrics, filename):
        pd.DataFrame(metrics).to_csv(self._result_path(filename))

    @property
    def _dem_rep_field(self):
        return "dem_rep"

    @property
    def posts_type(self):
        """
        Gets corresponding dataset folder name based on posts type
        """
        return self.post_type.posts_type(self)

    @property
    def submissions_dataset(self):
        """
        Dataset folder name for submissions
        """
        return "pushshift-reddit"

    @property
    def comments_dataset(self):
        """
        Dataset folder name for comments
        """
        return "pushshift-reddit-comments"
