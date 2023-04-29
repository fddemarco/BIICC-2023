from tqdm import tqdm
from pushshift_dataset import SubmissionsDataset
from pushshift_dataset import CommentsDataset
from file_handler import FileHandler
import pandas as pd


def process_monthly_dataset(dataset_cls, month, year, process):
    file_handler = FileHandler(dataset_cls(month, year))
    process(file_handler)


def run(dataset, years, months, process):
    for year in tqdm(years):
        for month in tqdm(months, leave=False):
            process_monthly_dataset(dataset, month, year, process)


def processing(handler):
    # handler.download()
    handler.process()
    # handler.delete_compressed()


if __name__ == '__main__':
    run(
        CommentsDataset,
        [2014],
        range(6, 7),
        processing
        )
