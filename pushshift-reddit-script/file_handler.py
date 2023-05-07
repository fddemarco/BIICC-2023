import os
import re
import string

import pandas as pd
import requests


class FileHandler:
    def __init__(self, pushshift_dataset):
        self.filename = pushshift_dataset.filename()
        self.pathname = pushshift_dataset.pathname()
        self.dir_pathname = pushshift_dataset.dir_pathname()
        self.url = pushshift_dataset.url()

        self.schema = pushshift_dataset.dataset_schema()
        self.text_features = pushshift_dataset.text_features()

    def download(self):

        headers = {'User-Agent': 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        with requests.get(self.url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(self.pathname_of_compressed(), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)



    def process(self):
        filename = self.pathname_of_compressed()
        i = 0
        for i, df in enumerate(
                pd.read_json(
                    filename,
                    compression={"method": "zstd", "max_window_size": 2 ** 31},
                    lines=True,
                    chunksize=10000, dtype=self.schema
                )
        ):
            df = df[self.schema.keys()]
            df = self.clean(df)
            self.to_parquet(df)

    def to_parquet(self, df):
        if os.path.exists(self.parquet_pathname()):
            df.to_parquet(self.parquet_pathname(), engine="fastparquet", append=True, index=False)
        else:
            df.to_parquet(self.parquet_pathname(), engine="fastparquet", index=False)

    def clean(self, df):
        punctuation_str = re.escape(string.punctuation)
        punctuation_regex = re.compile(f"([{punctuation_str}])")
        whitespaces_regex = re.compile(r"\s+")
        deleted_regex = re.compile(r'^\[(removed|deleted)]')

        for key in self.text_features:
            df.loc[:, key] = (df[key].fillna('')
                              .str.replace(deleted_regex, '', regex=True)
                              .str.replace(punctuation_regex, r" \1 ", regex=True)
                              .str.replace(whitespaces_regex, ' ', regex=True)
                              .str.strip().str.lower())
        df = df[df[self.text_features].ne('').any(axis=1)]
        return df

    def delete_compressed(self):
        if os.path.exists(self.pathname_of_compressed()):
            os.remove(self.pathname_of_compressed())

    def parquet_pathname(self):
        return os.path.join(self.dir_pathname, self.parquet_filename())

    def parquet_filename(self):
        return f"{self.filename}.parquet"

    def pathname_of_compressed(self):
        return self.pathname + '.zst'

