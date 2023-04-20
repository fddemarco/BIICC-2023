import os

import datasets as ds
import pandas as pd

from paths import DATA_DIR


class CommentsIterator:
    def __init__(self, subreddit, year, word_count, dataset='RC'):
        self.subreddit = subreddit
        self.year = year
        self.word_count = word_count
        self.dataset = dataset
        self.iterator = None

    def __iter__(self):
        data = ds.load_dataset('parquet',
                               data_files=self.data_files(),
                               split='train',
                               streaming=True)

        subreddit_data = data.filter(self.comments_by_subreddit,
                                     batched=True,
                                     batch_size=10000)
        self.iterator = subreddit_data
        return self

    def data_files(self):
        return os.path.join(DATA_DIR, f'{self.dataset}_{self.year}*')

    def comments_by_subreddit(self, dic):
        df = pd.DataFrame(dic)
        df["filter"] = df["subreddit"] == self.subreddit
        return df["filter"]

    def __next__(self):
        if self.iterator is None:
            raise ValueError("Iterator has not been initialized yet. Call iter() before calling next().")
        if self.word_count <= 0:
            raise StopIteration
        chunk_iterator = self.iterator.take(1000)
        self.iterator = self.iterator.skip(1000)
        return self.filter_chunk(chunk_iterator)

    def filter_chunk(self, current_chunk):
        df = pd.DataFrame(current_chunk)
        df['split'] = df['body'].str.split()  # TODO: no hardcodear body
        df['len'] = df['split'].str.len()
        total_len = df['len'].sum()
        if self.word_count < total_len:
            for row in df.itertuples():
                self.word_count -= row.len
                if self.word_count <= 0:
                    return df[:row.Index]
        else:
            self.word_count -= total_len
            return df
