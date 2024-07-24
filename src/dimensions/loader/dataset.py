import re

import huggingface_hub as hf_hub
import pathlib
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd


class PushshiftRedditPostsDataset:
    def __init__(self, year, working_directory):
        """
        Represents a dataset of Reddit posts for a given YEAR.

        :param year: Year of the posts (a value between 2012 and 2018)
        """
        self.year = str(year)
        self.working_dir = pathlib.Path(working_directory)

    @property
    def dataset_id(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def huggingface_dataset(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def schema(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    @property
    def text_features(self):
        raise NotImplementedError("Should be implemented in a subclass.")

    def name(self, month):
        return f"{self.dataset_id}_{self.year}-{month}"

    def download_data(self):
        hf_hub.snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            allow_patterns=self.hf_filename_pattern,
            local_dir=self.hf_data_dir,
            cache_dir=self.cache_dir,
            local_dir_use_symlinks=True
        )

    @property
    def hf_filename_pattern(self):
        return f"RS_{self.year}-[0-9]*.parquet"

    @property
    def repo_id(self):
        return "Aoppenhiem/pushshift-reddit"

    @property
    def cache_dir(self):
        return self.working_dir / 'cache'

    def clean_dataset(self):
        self.create_dir_structure()
        for month in self.months:
            self.clean_monthly_dataset(month)

    def create_dir_structure(self):
        if not self.clean_dir.exists():
            self.clean_dir.mkdir(parents=True)

    def clean_monthly_json_dataset(self, month):
        with pd.read_json(
                self.hf_pathname(month),
                lines=True,
                chunksize=100000,
                dtype=self.schema,
                dtype_backend="pyarrow") as reader:
            with pq.ParquetWriter(self.clean_pathname(month), self.schema, version="2.6") as writer:
                for chunk in reader:
                    chunk = chunk[self.schema.names].copy()
                    df = self.clean_batch(chunk)
                    table = pa.Table.from_pandas(df, schema=self.schema)
                    writer.write_table(table)

    def clean_monthly_dataset(self, month):
        data = ds.dataset(self.hf_pathname(month), format="parquet", schema=self.schema)
        with pq.ParquetWriter(self.clean_pathname(month), self.schema, version="2.6") as writer:
            for batch in data.to_batches(columns=self.schema.names):
                df = self.clean_batch(batch.to_pandas())
                table = pa.Table.from_pandas(df, schema=self.schema)
                writer.write_table(table)

    @property
    def months(self):
        return [str(i).zfill(2) for i in range(1, 12 + 1)]

    def hf_pathname(self, month):
        return self.hf_data_dir / 'data' / f'RS_{self.year}-{month}.parquet'

    def data_dir(self, base_folder):
        return self.working_dir / base_folder / self.huggingface_dataset / self.year

    @property
    def hf_data_dir(self):
        return self.data_dir('data')

    def clean_pathname(self, month):
        return self.clean_dir / f"{self.name(month)}_00.parquet"

    @property
    def clean_dir(self):
        return self.data_dir('data_clean')

    def clean_batch(self, df):
        regex = r"\[[^)]*]|(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)|^rt|http.+?"

        for key in self.text_features:
            df.loc[:, key] = (df[key].fillna('')
                              .str.replace(regex, '', regex=True)
                              .str.lower())
        return df[df[self.text_features].ne('').any(axis=1)]


class Submissions(PushshiftRedditPostsDataset):
    @property
    def dataset_id(self):
        return 'RS'

    @property
    def huggingface_dataset(self):
        return 'pushshift-reddit'

    @property
    def schema(self):
        return pa.schema(
            [
                ('author', pa.string()),
                ('created_utc', pa.string()),  # TODO: debería ser int64 segun paper de waller
                ('id', pa.string()),
                ('num_comments', pa.int64()),
                ('score', pa.int64()),
                ('selftext', pa.string()),
                ('subreddit', pa.string()),
                ('subreddit_id', pa.string()),
                ('title', pa.string())
            ])

    @property
    def text_features(self):
        return ["selftext", "title"]


class Comments(PushshiftRedditPostsDataset):
    @property
    def dataset_id(self):
        return 'RC'

    @property
    def huggingface_dataset(self):
        return 'pushshift-reddit-comments'

    @property
    def schema(self):
        return pa.schema(
            [
                ('author', pa.string()),
                ('body', pa.string()),
                ('controversiality', pa.int64()),
                ('created_utc', pa.int64()),
                ('id', pa.string()),
                ('link_id', pa.string()),
                ('score', pa.int64()),
                ('subreddit', pa.string()),
                ('subreddit_id', pa.string())
            ])

    @property
    def text_features(self):
        return ["body"]
