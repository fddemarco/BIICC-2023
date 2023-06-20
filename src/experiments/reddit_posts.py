import pandas as pd
import pyarrow.dataset as ds


class RedditPosts:
    def __init__(self, dataset, env, post_type):
        self.dataset = dataset
        self.env = env
        self.post_type = post_type

    @property
    def subreddit_field(self):
        return 'subreddit'

    @property
    def text_field(self):
        return 'text'

    def generate_text(self):
        splits = split_subreddits(self.dataset)

        for split in splits:
            grouped = self.texts_by_subreddit(
                ds.field(self.subreddit_field).isin(split)
            )
            text = grouped[self.text_field].str.cat(sep='\n')
            self.env.write_text(text)
        return splits

    def get_most_popular_subreddits(self):
        subreddits = get_most_popular_subreddits(self.dataset)
        return subreddits.index

    def texts_by_subreddit(self, filter_condition):
        filtered_dataset = self.dataset.filter(filter_condition)
        df = filtered_dataset.to_table().to_pandas()
        return self.texts_by_subreddit_df(df)

    def texts_by_subreddit_df(self, df):
        self.add_text_column(df)
        grouped = (
            df
            .groupby(self.subreddit_field)[self.text_field]
            .apply(lambda x: ' '.join(x))
            .reset_index()
           )
        return grouped

    def embeddings_for(self, model):
        embeddings = []
        ranked_subreddits, subreddits = self.get_ranked_subreddits()
        df = self.texts_by_subreddit(ds.field(self.subreddit_field).isin(ranked_subreddits))
        for s in ranked_subreddits:
            df_f = df[df["subreddit"] == s].copy()
            text = df_f["text"].str.cat(sep=' ')
            embeddings.append(model.get_sentence_vector(text).astype(float))
        return pd.DataFrame(embeddings, index=ranked_subreddits, columns=range(0, 300))

    def get_ranked_subreddits(self):
        subreddits = get_most_popular_subreddits(self.dataset)
        ranked_subreddits = get_ranked_subreddits_from(subreddits)
        return ranked_subreddits

    def add_text_column(self, df):
        df[self.text_field] = self.texts_from(df)

    def texts_from(self, df):
        return self.post_type.texts_from(self, df)

    def submissions_text(self, df):
        return df['title'] + ' ' + df['selftext']

    def comments_text(self, df):
        return df['body']


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


def split_subreddits(dataset):
    splits = []
    current_split = []
    current_cum_count = 0
    subreddits = get_most_popular_subreddits(dataset)
    threshold = partition_threshold(subreddits)
    for s, s_count in subreddits.items():
        current_split.append(s)
        current_cum_count += s_count
        if threshold < current_cum_count:
            splits.append(current_split)
            current_cum_count = 0
            current_split = []

    if current_split:
        splits.append(current_split)

    return splits


def get_ranked_subreddits_from(ranking):
    return [s for s in waller_ranking_arxiv() if s in ranking]


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
