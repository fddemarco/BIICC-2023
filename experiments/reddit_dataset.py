import os
from glob import glob

import pandas as pd
from datasets import load_dataset

from append_files import AppendFiles


class RedditDataset:
    def __init__(self, year, texts_dir, data_dir, results_dir):
        self.year = year
        self.texts_dir = texts_dir
        self.data_dir = data_dir
        self.results_dir = results_dir

    def write_to_file(self, df_chunk):
        df = pd.DataFrame(df_chunk)
        df['text'] = self.texts_from(df)
        grouped = df.groupby('subreddit')['text'].apply(lambda x: ' '.join(x)).reset_index()
        for idx, row in grouped.iterrows():
            with open(self.subreddit_pathname(row['subreddit'], '.txt'), 'a') as f:
                f.write(row['text'])
        return df_chunk

    def texts_by_subreddit(self):
        data = load_dataset(
            'parquet',
            data_files=self.data_files_pathname(),
            split='train',
            streaming=True
        )
        data_mapped = data.map(self.write_to_file, batched=True, batch_size=10000)
        for data in data_mapped:
            pass

    def data_files_pathname(self, ):
        return os.path.join(self.data_dir, f'{self.dataset()}_{self.year}*')

    def texts_to_single_file(self):
        files = glob(self.subreddit_pathname('', '*'))
        files.sort()
        with AppendFiles(files, self.subreddits_pathname('.txt')) as append_files:
            append_files.run()

    def subreddits_pathname(self, extension):
        return os.path.join(self.results_dir, self.subreddits_filename(extension))

    def subreddits_filename(self, extension):
        return f'subreddits_{self.year}' + extension

    def subreddit_pathname(self, subreddit, suffix):
        return os.path.join(self.texts_dir, f'subreddit_{self.year}_{subreddit}' + suffix)

    def dataset(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def texts_from(self, df):
        raise NotImplementedError("This method should be implemented in a subclass.")


class CommentsDataset(RedditDataset):
    def dataset(self):
        return 'RC'

    def texts_from(self, df):
        return df['body']


class SubmissionsDataset(RedditDataset):
    def dataset(self):
        return 'RS'

    def texts_from(self, df):
        return df['title'] + ' ' + df['selftext']
