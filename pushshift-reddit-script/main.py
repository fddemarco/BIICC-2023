from file_handler import FileHandler
from tqdm import tqdm
from pushshift_dataset import PushshiftDataset
from file_handler import push_to_hub
from file_handler import to_parquet


COMMENTS = 'comments'
SUBMISSIONS = 'submissions'


def process_monthly_dataset(dataset, month, year, process):
    monthly_dataset = PushshiftDataset(dataset, month, year)
    file = FileHandler(monthly_dataset)
    process(file)


def run(dataset, years, months, process):
    for year in tqdm(years):
        for month in tqdm(months, leave=False):
            process_monthly_dataset(dataset, month, year, process)


if __name__ == '__main__':
    run(
        SUBMISSIONS,
        [2006, 2007],
        [2, 3],
        (lambda file: (
            file.download(),
            file.decompress(),
            file.reduce_data(),
            to_parquet(file.small_format())
        ))
    )
