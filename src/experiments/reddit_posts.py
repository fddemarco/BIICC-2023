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

    @property
    def score_field(self):
        return 'score'

    def grouped_data_iterator(self):
        splits = self.split_subreddits()
        for split in splits:
            yield self.get_texts_by_subreddit(split)

    def grouped_split_iterator(self, split):
        group = self.get_texts_by_subreddit(split)
        yield from self.subreddit_iterator(group, split)

    def subreddit_iterator(self, group, split):
        for s in split:
            df = group[group[self.subreddit_field] == s].copy()
            yield df

    def posts_split_iterator(self, split):
        group = self.get_posts_in(split)
        yield from self.subreddit_iterator(group, split)

    def get_texts_by_subreddit(self, split):
        df = self.get_posts_in(split)
        group = self.texts_by_subreddit(df)
        return group

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
            .to_table(columns=[self.subreddit_field])
            .column(self.subreddit_field)
            .to_pandas()
            .value_counts()
            .sort_values(ascending=True)[-k:]
        )
        return subreddits

    def truncate_dataset(self, text_len_threshold=10000):
        dfs = []
        for split in self.split_subreddits():
            for df in self.posts_split_iterator(split):
                df_s = self.truncate_subreddit(df, text_len_threshold)
                dfs.append(df_s)
        result = pd.concat(dfs).reset_index(drop=True)
        return result

    def truncate_subreddit(self, df, text_len_threshold):
        df.sort_values(self.score_field, ascending=False)
        text_len = df[self.text_field].str.len()
        df = df[text_len < text_len_threshold]

        cumulative_len = df[self.text_field].str.len().cumsum()
        df = df[cumulative_len < text_len_threshold]
        return df

    def generate_text(self):
        for split_group in self.grouped_data_iterator():
            text = split_group[self.text_field].str.cat(sep='\n')
            self.sink.write_text(text)

    def get_posts_in(self, subreddits):
        filter_condition = ds.field(self.subreddit_field).isin(subreddits)
        filtered_dataset = self.dataset.filter(filter_condition)
        df = filtered_dataset.to_table().to_pandas()
        df[self.text_field] = self.texts_from(df)
        return df

    def texts_by_subreddit(self, df):
        grouped = (
            df
            .groupby(self.subreddit_field)[self.text_field]
            .apply(lambda x: ' '.join(x))
            .reset_index()
           )
        return grouped

    def generate_embeddings_for(self, model):
        ranked_subreddits = self.get_ranked_subreddits()
        embeddings = self.embeddings_for_subreddits(model, ranked_subreddits)
        return pd.DataFrame(embeddings, index=ranked_subreddits, columns=range(0, 300))

    def embeddings_for_subreddits(self, model, subreddits):
        embeddings = []
        for df in self.grouped_split_iterator(subreddits):
            embeddings.append(self.embedding_for(df, model))
        return embeddings

    def embedding_for(self, df, model):
        text = df[self.text_field].item()
        embedding = model.get_sentence_vector(text).astype(float)
        return embedding

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
    return int(subreddits.sum()) // 20


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
