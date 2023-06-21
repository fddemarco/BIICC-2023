import pandas as pd
import pyarrow.dataset as ds

from posts_type import Submissions, Comments


class RedditPosts:
    @classmethod
    def from_submissions(cls, dataset, env):
        return cls(dataset, env, Submissions())

    @classmethod
    def from_comments(cls, dataset, env):
        return cls(dataset, env, Comments())

    def __init__(self, dataset, env, post_type):
        self.dataset = dataset
        self.sink = env
        self.post_type = post_type

    @property
    def subreddit_field(self):
        return 'subreddit'

    @property
    def text_field(self):
        return 'text'

    @property
    def title_field(self):
        return 'title'

    @property
    def selftext_field(self):
        return 'selftext'

    @property
    def body_field(self):
        return 'body'

    def split_subreddits(self):
        splits = []
        current_split = []
        current_cum_count = 0
        subreddits = self.get_most_popular_subreddits()
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

    def get_most_popular_subreddits(self, k=10000):
        """
        Returns top 10k subreddits with more posts, sorted in ascending order.
        """
        subreddits = (
            self.dataset
            .to_table(columns=["subreddit"])
            .column("subreddit")
            .to_pandas()
            .value_counts()
            .sort_values(ascending=True)[-k:]
        )
        return subreddits

    def generate_text(self):
        splits = self.split_subreddits()
        for split in splits:
            grouped = self.get_posts_for_subreddits_in(split)
            text = grouped[self.text_field].str.cat(sep='\n')
            self.sink.write_text(text)
        return splits

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

    def add_text_column(self, df):
        df[self.text_field] = self.texts_from(df)

    def generate_embeddings_for(self, model):
        ranked_subreddits = self.get_ranked_subreddits()
        embeddings = self.embeddings_for_subreddits(model, ranked_subreddits)
        return pd.DataFrame(embeddings, index=ranked_subreddits, columns=range(0, 300))

    def embeddings_for_subreddits(self, model, subreddits):
        embeddings = []
        df = self.get_posts_for_subreddits_in(subreddits)
        for s in subreddits:
            df_f = df[df[self.subreddit_field] == s].copy()
            embeddings.append(self.embedding_for_subreddit(df_f, model))
        return embeddings

    def embedding_for_subreddit(self, df, model):
        text = df[self.text_field].str.cat(sep=' ')
        embedding = model.get_sentence_vector(text).astype(float)
        return embedding

    def get_posts_for_subreddits_in(self, ranked_subreddits):
        return self.texts_by_subreddit(ds.field(self.subreddit_field).isin(ranked_subreddits))

    def get_ranked_subreddits(self):
        subreddits = self.get_most_popular_subreddits()
        ranked_subreddits = get_ranked_subreddits_from(subreddits)
        return ranked_subreddits

    def texts_from(self, df):
        return self.post_type.texts_from(self, df)

    def submissions_text(self, df):
        return (df[self.title_field]
                .str.cat(df[self.selftext_field], sep=' ')
                .str.strip())

    def comments_text(self, df):
        return df[self.body_field]


def partition_threshold(subreddits):
    return int(subreddits.sum()) // 100


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
