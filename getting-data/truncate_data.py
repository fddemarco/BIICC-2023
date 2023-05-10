import os
from glob import glob

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


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


if __name__ == "__main__":
    data_dir = input("Enter data dir: ")
    results_dir = input("Enter results dir: ")
    year = 2012
    unique_subreddits = ["democrats"]
    subreddit_texts = []
    subreddit_cant = []
    for subR in unique_subreddits:
        dfR = truncate_data("RS", year, subR, 10000, data_dir)
        output_pathname = os.path.join(results_dir, "democrats.csv")
        dfR.to_csv(output_pathname, index=False)

        subreddit_texts.append(' '.join(dfR['title'] + ' ' + dfR['text']).replace('\r', '').replace('\n', ''))
        subreddit_cant.append(dfR.shape[0])
