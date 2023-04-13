import os
from reddit_data_processor import SubmissionsProcessor
from reddit_data_processor import CommentsProcessor


class PushshiftDataset:
    def __init__(self, dataset, month, year):
        self.valid_dataset(dataset)
        self.valid_month(month)
        self.valid_year(year)
        self.dataset = dataset
        self.month = month_to_str(month)
        self.year = str(year)
        self.base_local_dir = '/media/franco/disco/BIICC/git-lfs/'

    def value(self):
        if self.dataset == 'submissions':
            return 'RS'
        elif self.dataset == 'comments':
            return 'RC'

    def pushshift_path(self):
        return f"{self.value()}_{self.year}-{self.month}"

    def url(self):
        return f"https://files.pushshift.io/reddit/{self.dataset}/{self.pushshift_path()}.zst"

    def build_local_path(self):
        return os.path.join(self.base_local_dir, self.dataset_folder(), 'data')

    def dataset_folder(self):
        if self.dataset == 'submissions':
            return 'pushshift-reddit'
        elif self.dataset == 'comments':
            return 'pushshift-reddit-comments'

    def data_processor(self):
        if self.dataset == 'submissions':
            return SubmissionsProcessor(self)
        elif self.dataset == 'comments':
            return CommentsProcessor(self)

    @staticmethod
    def valid_dataset(name):
        if name not in ['submissions', 'comments']:
            raise ValueError(f'Invalid argument ({name}). Please provide either "submissions" or "comments".')

    @staticmethod
    def valid_month(month):
        if month not in range(1, 12+1):
            raise ValueError(f'Invalid argument ({month}). Please provide a valid month (1-12).')

    @staticmethod
    def valid_year(year):
        if year not in range(2006, 2018+1):
            raise ValueError(f'Invalid argument ({year}). Please provide a valid year (2006-2018).')


def month_to_str(month):
    return str(month).zfill(2)
