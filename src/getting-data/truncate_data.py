import numpy as np

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa


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


def split_subreddits(subreddits):  # Subset sum
    splits = []
    current_split = []
    current_cum_count = 0
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


def write_splits(dataset, splits, base_dir):
    for idx, ss in enumerate(splits):
        dataset_filtered = dataset.filter(ds.field("subreddit").isin(ss))
        write_options = ds.ParquetFileFormat().make_write_options(version="2.6")
        ds.write_dataset(
            dataset_filtered,
            base_dir=base_dir,
            basename_template=f"part-{{i}}{idx}.parquet",
            file_options=write_options,
            format="parquet",
            existing_data_behavior="overwrite_or_ignore"
        )


def truncate_splits(splits, base_dir, text_len_threshold):
    dfs = []
    for i, ss in enumerate(splits):
        dataset = ds.dataset(f"{base_dir}/part-0{i}.parquet", format="parquet")
        for s in ss:
            dataset_s = dataset.filter(ds.field("subreddit") == s)
            df_s = truncate_data(dataset_s, text_len_threshold)
            dfs.append(df_s)
    result = pd.concat(dfs).reset_index(drop=True)
    return result


def truncate_data(dataset, text_len_threshold):
    df = dataset.to_table().to_pandas()
    df.sort_values("score", ascending=False)
    df["text"] = df["title"] + " " + df["selftext"]
    df["cumulative_len"] = df["text"].str.len().cumsum()
    first_row = np.zeros(len(df), dtype=bool)
    first_row[0] = True
    df = df[
        (df["cumulative_len"] <= text_len_threshold) | first_row
        ]
    return df


def get_data():
    data_dir = input("Enter data dir: ")
    results_dir = input("Enter results dir: ")
    partitioning = ds.partitioning(
        pa.schema([
            ("month", pa.int16())
        ])
    )
    dataset = ds.dataset(data_dir, format="parquet", partitioning=partitioning)
    subreddits = get_most_popular_subreddits(dataset)
    splits = split_subreddits(subreddits)
    write_splits(dataset, splits, results_dir)
    df = truncate_splits(splits, results_dir, 10000)
    df.to_parquet(f"{results_dir}/results.parquet")


if __name__ == "__main__":
    get_data()
