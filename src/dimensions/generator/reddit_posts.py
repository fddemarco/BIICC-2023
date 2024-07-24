from typing import Iterator, List, TypeAlias

import pandas as pd
import pyarrow.dataset as ds

from dimensions.generator.posts_type import Comments, Submissions

Community: TypeAlias = str


class RedditPosts:
    @classmethod
    def from_submissions(cls, dataset, sink):
        return cls(dataset, sink, Submissions())

    @classmethod
    def from_comments(cls, dataset, sink):
        return cls(dataset, sink, Comments())

    def __init__(self, dataset, sink, post_type):
        self.dataset = dataset
        self.sink = sink
        self.post_type = post_type

    @property
    def subreddit_field(self):
        return "subreddit"

    @property
    def text_field(self):
        return "text"

    @property
    def title_field(self):
        return "title"

    @property
    def selftext_field(self):
        return "selftext"

    @property
    def body_field(self):
        return "body"

    @property
    def score_field(self):
        return "score"

    def grouped_data_iterator(self) -> Iterator[pd.DataFrame]:
        """Iterate over community splits text.

        Yields all text content of a community split at a time.

        Yields:
            pd.DataFrame: Dataframe of all text content concatenated for each community.
        """
        splits = self.split_subreddits()
        for split in splits:
            yield self.get_texts_by_subreddit(split)

    def split_data_iterator(self) -> Iterator[pd.DataFrame]:
        """Iterate over all communities text.

        Yields all text content of a single community at a time.

        Yields:
            Iterator[pd.DataFrame]: Single community text data.
        """
        splits = self.split_subreddits()
        for split in splits:
            yield from self.grouped_split_iterator(split)

    def grouped_split_iterator(self, split: List[Community]) -> Iterator[pd.DataFrame]:
        """Iterate over community text within a certain split.

            Yields all text content of each community at a time.

        Args:
            split (List[Community]): List of communities to be iterated

        Yields:
            Iterator[pd.DataFrame]: All text content of a community (concatenated posts)
        """
        grouped_text = self.get_texts_by_subreddit(split)
        yield from self.subreddit_iterator(grouped_text, split)

    def posts_split_iterator(self, split: List[Community]) -> Iterator[pd.DataFrame]:
        """Iterate over community posts within a certain split.

        Yields a all posts of a community at a time.

        Args:
            split (List[Community]): List of communities to be iterated

        Yields:
            Iterator[pd.DataFrame]: All posts of a community (individual posts)
        """
        group = self.get_posts_in(split)
        yield from self.subreddit_iterator(group, split)

    def subreddit_iterator(
        self, data: pd.DataFrame, split: List[Community]
    ) -> Iterator[pd.DataFrame]:
        """Iterate over communities data in a split

        Args:
            data (pd.DataFrame): Input data
            split (List[Community]): List of communities to be iterated

        Yields:
            Iterator[pd.DataFrame]: Data from a single community
        """
        for s in split:
            filter_condition = data[self.subreddit_field] == s
            yield data[filter_condition].copy()

    def get_texts_by_subreddit(self, split):
        df = self.get_posts_in(split)
        return self.texts_by_subreddit(df)

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
            self.dataset.to_table(columns=[self.subreddit_field])
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
        text_len = df[self.text_field].str.len()  # No funciona correctamente
        df = df[
            text_len < text_len_threshold
        ]  # porque no toma en cuenta los espacios del join

        cumulative_len = df[self.text_field].str.len().cumsum()
        df = df[cumulative_len < text_len_threshold]
        return df

    def generate_text(self):
        for split_group in self.grouped_data_iterator():
            text = split_group[self.text_field].str.cat(sep="\n")
            self.sink.write_text(text)

    # Falta ignorar aquellos comentarios sin texto
    def get_posts_in(self, subreddits):
        filter_condition = ds.field(self.subreddit_field).isin(subreddits)
        filtered_dataset = self.dataset.filter(filter_condition)
        df = filtered_dataset.to_table().to_pandas()
        df[self.text_field] = self.texts_from(df)
        return df

    def texts_by_subreddit(self, df):
        grouped = (
            df.groupby(self.subreddit_field)[self.text_field]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )
        return grouped

    def generate_embeddings_for(self, model):
        embeddings = []
        subreddits = []
        for df in self.split_data_iterator():
            embedding, subreddit = self.embedding_for(df, model)
            embeddings.append(embedding)
            subreddits.append(subreddit)
        return pd.DataFrame(embeddings, index=subreddits, columns=range(0, 300))

    def embedding_for(self, df, model):
        text = df[self.text_field].item()
        embedding = model.get_sentence_vector(text).astype(float)
        subreddit = df[self.subreddit_field].item()
        return embedding, subreddit

    def texts_from(self, df):
        return self.post_type.texts_from(self, df)

    def submissions_text(self, df):
        return (
            df[self.title_field].str.cat(df[self.selftext_field], sep=" ").str.strip()
        )

    def comments_text(self, df):
        return df[self.body_field]


def partition_threshold(subreddits):
    return int(subreddits.sum()) // 100
