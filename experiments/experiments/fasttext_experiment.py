import fasttext
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
import pandas as pd
import pathlib

import experiments.dimension_generator as dg
import experiments.reddit_posts as ps
import experiments.ranking as rk


def mkdir_if_not_exists(_dir):
    if not _dir.exists():
        _dir.mkdir(parents=True)
    return _dir


class FasttextExperiment:
    def __init__(self, year, working_dir, results_dir, post_type, dataset):
        self.year = str(year)
        self.working_dir = pathlib.Path(working_dir)
        self.results_folder = str(results_dir)
        self.post_type = post_type
        self.dataset = str(dataset)

    @property
    def base_dataset_dir(self):
        return self.base_parent_dir / self.dataset

    @property
    def base_parent_dir(self):
        return self.working_dir / self.dataset_type / self.year

    @property
    def data_pathname(self):
        return self.base_dataset_dir / 'data'

    @property
    def truncated_data_pathname(self):
        truncated_dir = self.base_parent_dir / 'truncated'
        return mkdir_if_not_exists(truncated_dir)

    @property
    def results_dir(self):
        results_dir = self.base_dataset_dir / self.results_folder
        return mkdir_if_not_exists(results_dir)

    @property
    def subreddits_pathname(self):
        return self.results_dir / 'subreddits.txt'

    @property
    def fasttext_model_pathname(self):
        return self.results_dir / 'subreddits.bin'

    def embedding_pathname(self):
        return self.results_dir / 'embeddings.csv'

    def generate_texts(self):
        reddit_posts = self.get_reddit_posts()
        reddit_posts.generate_text()

    def generate_truncated_texts(self):
        reddit_posts = self.get_reddit_posts()
        df = reddit_posts.truncate_dataset()
        df.to_parquet(self.truncated_data_pathname / 'truncated_data.parquet')

    def get_reddit_posts(self):
        dataset = ds.dataset(self.data_pathname, format="parquet")
        reddit_posts = ps.RedditPosts(dataset, self, self.post_type)
        return reddit_posts

    def write_text(self, text):
        with open(self.subreddits_pathname, 'a') as f:
            f.write(text + '\n')

    def save_embeddings_to_csv(self):
        model_pathname = str(self.fasttext_model_pathname.absolute())
        model = fasttext.load_model(model_pathname)
        reddit_posts = self.get_reddit_posts()

        tf_idf = reddit_posts.generate_embeddings_for(model)
        tf_idf.to_csv(self.embedding_pathname())

    def get_fasttext_scores(self):
        df = pd.read_csv(self.embedding_pathname(), index_col=0)
        dimensions = dg.DimensionGenerator(df).generate_dimensions_from_seeds([("democrats", "Conservative")])
        scores = dg.score_embedding(df, zip([self.dem_rep_field], dimensions))
        return scores

    def compare_rankings(self):
        fasttext_ranking = self.get_fasttext_scores().to_dict()[self.dem_rep_field]
        ranking = rk.Ranking(fasttext_ranking)
        metrics = ranking.compare_ranking()

        pd.DataFrame(metrics.classification_metrics).to_csv(self.results_dir / 'classification_metrics.csv')
        pd.DataFrame(metrics.ranking_metrics).to_csv(self.results_dir / 'ranking_metrics.csv')

        for name, fig in metrics.plots.items():
            fig.savefig(
                self.results_dir / f"{name}.png",
                dpi=300,
                bbox_inches='tight')
            plt.close(fig)

    @property
    def dem_rep_field(self):
        return 'dem_rep'

    @property
    def dataset_type(self):
        return self.post_type.dataset_type(self)

    @property
    def submissions_dataset(self):
        return 'pushshift-reddit'

    @property
    def comments_dataset(self):
        return 'pushshift-reddit-comments'
