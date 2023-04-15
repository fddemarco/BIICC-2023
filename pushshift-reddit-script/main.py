from tqdm import tqdm
from pushshift_dataset import SubmissionsDataset
from pushshift_dataset import CommentsDataset
from file_handler import push_to_hub
from file_handler import to_parquet


def process_monthly_dataset(dataset_cls, month, year, process):
    dataset = dataset_cls(month, year).file_handler()
    process(dataset)


def run(dataset, years, months, process):
    for year in tqdm(years):
        for month in tqdm(months, leave=False):
            process_monthly_dataset(dataset, month, year, process)


if __name__ == '__main__':
    run(
        SubmissionsDataset,
        [2012],
        [1],
        (lambda file: (
            file.download(),
            file.decompress(),
            file.reduce_data(),
            to_parquet(file.small_format()),
            push_to_hub(file.parquet_format())
        ))
    )
