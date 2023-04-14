from file_handler import FileHandler
from tqdm import tqdm
from pushshift_dataset import PushshiftDataset
from file_handler import push_to_hub
from file_handler import to_parquet


COMMENTS = 'comments'
SUBMISSIONS = 'submissions'


class Pushshift:
    def __init__(
            self,
            dataset,
            years,
            months,
            file_operation
    ):
        self.dataset = dataset
        self.years = years
        self.months = months
        self.file_operation = file_operation

    def run(self):
        for year in tqdm(self.years):
            for month in tqdm(self.months, leave=False):
                self.process_monthly_dataset(month, year)

    def process_monthly_dataset(self, month, year):
        monthly_dataset = PushshiftDataset(self.dataset, month, year)
        file = FileHandler(monthly_dataset)
        self.file_operation(file)


if __name__ == '__main__':
    Pushshift(
        SUBMISSIONS,
        [2006, 2007],
        [2, 3],
        (lambda file: (
            file.download(),
            file.decompress(),
            file.reduce_data(),
            to_parquet(file.small_format())
        ))
    ).run()
