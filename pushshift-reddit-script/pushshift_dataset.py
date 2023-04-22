import os
from reddit_data_processor import SubmissionsProcessor
from reddit_data_processor import CommentsProcessor
from file_handler import FileHandler


class PushshiftDataset:
    def __init__(self, month, year):
        self.valid_input(month, year)
        self.base_local_dir = self.valid_environment()
        self.month = month_to_str(month)
        self.year = str(year)

    @staticmethod
    def valid_environment():
        repo_path = os.environ.get('HUGGINGFACE_DIR_PATH')
        help_message = (
            "Please set it to the absolute path of your git repository folder, e.g.: \n\n"
            "if your repository is at /path/to/your/git/repo, do "
            "export HUGGINGFACE_DIR_PATH=/path/to/your/git\n\n"
            "You can do this by adding the above command to your shell configuration file, "
            "such as .bashrc or .zshrc, or by running it in your terminal before running the script."
        )
        if repo_path is None:
            raise ValueError(
                f"The HUGGINGFACE_REPO_PATH environment variable is not set. "
                + help_message
                )
        elif not os.path.exists(repo_path):
            raise ValueError(
                f"The HUGGINGFACE_DIR_PATH={repo_path} environment variable is set to a non-existent directory. "
                + help_message
            )
        return repo_path



    def valid_input(self, month, year):
        self.valid_month(month)
        self.valid_year(year)

    def value(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def pushshift_path(self):
        return f"{self.value()}_{self.year}-{self.month}"

    def url(self):
        return f"https://files.pushshift.io/reddit/{self.dataset_url()}/{self.pushshift_path()}.zst"

    def dataset_url(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def build_local_path(self):
        return os.path.join(self.base_local_dir, self.dataset_folder(), 'data')

    def dataset_folder(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def data_processor(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def file_handler(self):
        return FileHandler(self)

    @staticmethod
    def valid_month(month):
        if month not in range(1, 12+1):
            raise ValueError(f'Invalid argument ({month}). Please provide a valid month (1-12).')

    @staticmethod
    def valid_year(year):
        if year not in range(2006, 2018+1):
            raise ValueError(f'Invalid argument ({year}). Please provide a valid year (2006-2018).')


class CommentsDataset(PushshiftDataset):
    def value(self):
        return 'RC'

    def data_processor(self):
        return CommentsProcessor(self.pushshift_path())

    def dataset_folder(self):
        return 'pushshift-reddit-comments'

    def dataset_url(self):
        return 'comments'


class SubmissionsDataset(PushshiftDataset):
    def value(self):
        return 'RS'

    def data_processor(self):
        return SubmissionsProcessor(self.pushshift_path())

    def dataset_folder(self):
        return 'pushshift-reddit'

    def dataset_url(self):
        return 'submissions'


def month_to_str(month):
    return str(month).zfill(2)
