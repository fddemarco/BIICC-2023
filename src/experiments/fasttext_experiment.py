import fasttext
import pyarrow.dataset as ds
import pandas as pd
import pathlib


import dimension_generator as dg
import reddit_posts as ps
import ranking as rk


class FasttextExperiment:
    def __init__(self, year, working_dir, results_dir, post_type):
        self.year = str(year)
        self.working_dir = pathlib.Path(working_dir)
        self.results_folder = results_dir
        self.post_type = post_type

    @property
    def base_dataset_dir(self):
        return self.working_dir / self.dataset_type / self.year

    @property
    def data_pathname(self):
        return self.base_dataset_dir / 'data'

    @property
    def truncated_data_pathname(self):
        truncated_dir = self.base_dataset_dir / 'truncated'
        if not truncated_dir.exists():
            truncated_dir.mkdir(parents=True)
        return truncated_dir

    @property
    def results_dir(self):
        results_dir = self.base_dataset_dir / self.results_folder
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        return results_dir

    @property
    def subreddits_pathname(self):
        return self.results_dir / 'subreddits.txt'

    @property
    def fasttext_model_pathname(self):
        return self.results_dir / 'subreddits.bin'

    def embedding_pathname(self):
        return self.results_dir / 'embeddings.csv'

    def generate_texts(self, pathname=None):
        reddit_posts = self.get_reddit_posts(pathname)
        reddit_posts.generate_text()

    def generate_truncated_texts(self):
        reddit_posts = self.get_reddit_posts()
        df = reddit_posts.truncate_dataset()
        df.to_parquet(self.truncated_data_pathname / 'truncated_data.parquet')

    def get_reddit_posts(self, pathname=None):
        if pathname is None:
            pathname = self.data_pathname
        dataset = ds.dataset(pathname, format="parquet")
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
        plot = ranking.bump_plot()
        plot.savefig(
            self.results_dir / f'rankings_comparison.png',
            dpi=300,
            bbox_inches='tight')

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
