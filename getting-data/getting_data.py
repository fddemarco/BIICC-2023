import os
from glob import glob

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from huggingface_hub import snapshot_download


class GetData:
    def __init__(self, year, local_dir=''):
        self.year = year
        self.local_dir = local_dir

    def download_data(self):
        snapshot_download(
            repo_id="fddemarco/pushshift-reddit",
            repo_type="dataset",
            allow_patterns=self.dataset_pattern(),
            local_dir=self.local_dir,
            local_dir_use_symlinks=False)

    def huggingface_dataset(self):
        raise NotImplementedError("Should be implemented in a subclass")

    def dataset_pattern(self):
        raise NotImplementedError("Should be implemented in a subclass")


class GetSubmissions(GetData):
    def huggingface_dataset(self):
        return "pushshift-reddit"

    def dataset_pattern(self):
        return f"data/{self.dataset}_{self.year}-[0-9]*_[0-9]*.parquet"

    @property
    def dataset(self):
        return "RS"


class GetComments(GetData):
    def huggingface_dataset(self):
        return "pushshift-reddit-comments"

    def dataset_pattern(self):
        return f"data/{self.dataset}_{self.year}-[0-9]*.parquet"

    @property
    def dataset(self):
        return "RC"


def truncate_data(dataset_source, year, subreddit, text_len_threshold, data_dir):
    """
        Ejemplo de uso:
            truncate_data("RS", 2012, "TwoXChromosomes", 100000)
    """
    data_files_pattern = os.path.join(data_dir, f"{dataset_source}_{year}-[0-9]*_[0-9]*.parquet")
    data_files = glob(data_files_pattern)

    dataset = ds.dataset(data_files)
    dataset = dataset.filter(pc.field("subreddit") == subreddit)
    sorted_dataset = dataset.sort_by([("score", "descending")])

    dfs = []
    for chunk in sorted_dataset.to_batches():
        df = chunk.to_pandas()
        df["text"] = df["title"] + " " + df["selftext"]
        df["cumulative_len"] = df["text"].str.len().cumsum()
        if text_len_threshold <= df["cumulative_len"].iloc[-1]:
            df = df[df["cumulative_len"] <= text_len_threshold]
            dfs.append(df)
            break
        else:
            dfs.append(df)
    result = pd.concat(dfs).reset_index(drop=True)
    return result
