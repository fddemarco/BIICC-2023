import numpy as np
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
import pandas as pd
import pathlib
import fasttext

import src.experiments.dimension_generator as dg


class FasttextExperiment:
    def __init__(self, year, working_dir, results_dir):
        self.year = str(year)
        self.working_dir = pathlib.Path(working_dir)
        self.results_folder = results_dir

    @property
    def base_dataset_dir(self):
        return self.working_dir / self.dataset_type / self.year

    @property
    def data_pathname(self):
        return self.base_dataset_dir / 'data'

    @property
    def results_dir(self):
        results_dir = self.base_dataset_dir / self.results_folder
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        return results_dir

    def generate_texts(self):
        dataset = ds.dataset(self.data_pathname, format="parquet")
        splits = split_subreddits(dataset)

        for split in splits:
            grouped = self.texts_by_subreddit(dataset, ds.field("subreddit").isin(split))
            text = grouped["text"].str.cat(sep="\n")
            with open(self.subreddits_pathname, 'a') as f:
                f.write(text + '\n')
        return splits

    def texts_by_subreddit(self, dataset, filter_condition):
        dataset_split = dataset.filter(filter_condition)
        df = dataset_split.to_table().to_pandas()
        return self.texts_by_subreddit_df(df)

    def texts_by_subreddit_df(self, df):
        self.add_text_column(df)
        grouped = df.groupby('subreddit')['text'].apply(lambda x: ' '.join(x)).reset_index()
        return grouped

    def add_text_column(self, df):
        df['text'] = self.texts_from(df)

    @property
    def subreddits_pathname(self):
        return self.results_dir / 'subreddits.txt'

    @property
    def fasttext_output_pathname(self):
        return self.results_dir / 'subreddits'

    @property
    def fasttext_model_pathname(self):
        return self.results_dir / 'subreddits.bin'

    def get_most_popular_subreddits(self):
        dataset = ds.dataset(self.data_pathname, format="parquet")
        subreddits = get_most_popular_subreddits(dataset)
        return subreddits.index

    def save_embeddings_to_csv(self):
        subreddits = self.get_most_popular_subreddits()
        subreddits = np.intersect1d(waller_ranking_arxiv(), subreddits)
        subreddits = list(subreddits)

        embeddings = self.embeddings_of(subreddits)
        tf_idf = pd.DataFrame(embeddings, index=subreddits, columns=range(0, 300))
        tf_idf.to_csv(self.embedding_pathname())
        return tf_idf

    def embedding_pathname(self):
        return self.results_dir / 'embeddings.csv'

    def embeddings_of(self, subreddits):
        model_pathname = str(self.fasttext_model_pathname.absolute())
        model = fasttext.load_model(model_pathname)
        embeddings = []
        dataset = ds.dataset(self.data_pathname, format="parquet")
        df = self.texts_by_subreddit(dataset, ds.field("subreddit").isin(subreddits))
        for s in subreddits:
            df_f = df[df["subreddit"] == s].copy()
            text = df_f["text"].str.cat(sep=' ')  # TODO: REVISAR
            embeddings.append(model.get_sentence_vector(text).astype(float))
        return embeddings

    def compare_rankings(self):
        fasttext_ranking = list(pd.read_csv(self.embedding_pathname(), index_col=0).index)
        waller_ranking = get_waller_ranking_for(fasttext_ranking)
        rankings = [
            {
                "Model": ["waller", "fasttext"],
                "Rank": [i + 1, fasttext_ranking.index(subreddit) + 1],
                "Subreddit": subreddit
            } for i, subreddit in enumerate(waller_ranking)
        ]
        bump_chart(rankings, len(waller_ranking), self.results_dir)

    def get_fasttext_ranking(self):
        scores = self.get_fasttext_scores()
        fasttext_ranking = [x for x in scores.sort_values('dem_rep').index]
        return fasttext_ranking

    def get_fasttext_scores(self):
        df = pd.read_csv(self.embedding_pathname(), index_col=0)
        dimensions = dg.DimensionGenerator(df).generate_dimensions_from_seeds([("democrats", "Conservative")])
        scores = dg.score_embedding(df, zip(["dem_rep"], dimensions))
        return scores

    @property
    def dataset_type(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def texts_from(self, df):
        raise NotImplementedError("This method should be implemented in a subclass.")


class FasttextExperimentForComments(FasttextExperiment):
    def texts_from(self, df):
        return df['body']

    @property
    def dataset_type(self):
        return 'pushshift-reddit-comments'


class FasttextExperimentForSubmissions(FasttextExperiment):
    def texts_from(self, df):
        return df['title'] + ' ' + df['selftext']

    @property
    def dataset_type(self):
        return 'pushshift-reddit'


def get_waller_ranking_for(ranking):
    return [s for s in waller_ranking_arxiv() if s in ranking]


def bump_chart(elements, n, results_dir):
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
    plt.yticks([i for i in range(1, n+1)])

    ax.set_xlabel('Model')
    ax.set_ylabel('Rank')
    ax.set_title('Comparison of Models on Subreddit Classification Task')

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(
        results_dir / f'rankings_comparison.png',
        dpi=300,
        bbox_inches='tight')


def waller_ranking_arxiv():
    return [
        'democrats',
        'EnoughLibertarianSpam',
        'hillaryclinton',
        'progressive',
        'BlueMidterm2018',
        'EnoughHillHate',
        'Enough_Sanders_Spam',
        'badwomensanatomy',
        'racism',
        'GunsAreCool',
        'Christians',
        'The_Farage',
        'new_right',
        'conservatives',
        'metacanada',
        'Mr_Trump',
        'NoFapChristians',
        'TrueChristian',
        'The_Donald',
        'Conservative'
    ]


def get_most_popular_subreddits(dataset):
    """
    Returns top 10k subreddits with more posts, sorted in ascending order.
    """
    subreddits = (
        dataset
        .to_table(columns=["subreddit"])
        .column("subreddit")
        .to_pandas()
        .value_counts()
        .sort_values(ascending=True)[-10000:]
    )
    return subreddits


def partition_threshold(subreddits):
    return int(subreddits.sum()) // 100


def split_subreddits(dataset):  # Subset sum
    splits = []
    current_split = []
    current_cum_count = 0
    subreddits = get_most_popular_subreddits(dataset)
    threshold = partition_threshold(subreddits)
    for s, s_count in subreddits.items():
        if threshold < current_cum_count + s_count:
            splits.append(current_split)
            current_cum_count = 0
            current_split = []
        current_cum_count += s_count
        current_split.append(s)

    if current_split:
        splits.append(current_split)

    return splits
