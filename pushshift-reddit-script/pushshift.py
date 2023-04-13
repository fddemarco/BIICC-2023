from file_handler import FileHandler
from tqdm import tqdm
from pushshift_dataset import PushshiftDataset
from file_handler import push_to_hub
from file_handler import to_parquet


class Pushshift:
    def __init__(self, dataset, start_year, end_year, start_month, end_month):
        self.dataset = dataset
        self.start_year = start_year
        self.end_year = end_year
        self.start_month = start_month
        self.end_month = end_month

    def run(self):
        for year in tqdm(range(self.start_year, self.end_year + 1)):
            for month in tqdm(range(self.start_month, self.end_month + 1), leave=False):
                self.process_monthly_dataset(month, year)

    def process_monthly_dataset(self, month, year):
        monthly_dataset = PushshiftDataset(self.dataset, month, year)
        file = FileHandler(monthly_dataset)
        file.download()
        file.decompress()
        file.reduce_data()
        to_parquet(file.small_format())
        push_to_hub(file.parquet_format())


if __name__ == '__main__':
    Pushshift('comments', 2006, 2006, 1, 1).run()
