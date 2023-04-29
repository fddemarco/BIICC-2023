import os


class PushshiftDataset:
    def __init__(self, month, year):
        self.valid_input(month, year)

        self.month = month_to_str(month)
        self.year = str(year)
        self.base_local_dir = None

        self.valid_environment()

    def valid_input(self, month, year):
        self.valid_month(month)
        self.valid_year(year)

    @staticmethod
    def valid_month(month):
        if month not in range(1, 12+1):
            raise ValueError(f'Invalid argument ({month}). Please provide a valid month (1-12).')

    @staticmethod
    def valid_year(year):
        if year not in range(2006, 2018+1):
            raise ValueError(f'Invalid argument ({year}). Please provide a valid year (2006-2018).')

    def valid_environment(self):
        repo_path = os.environ.get('HUGGINGFACE_DIR_PATH')
        self.assert_path_variable_is_set(repo_path)
        self.assert_path_variable_is_valid(repo_path)

        self.base_local_dir = repo_path

        pathname_dir = self.dir_pathname()
        if not os.path.exists(pathname_dir):
            os.makedirs(pathname_dir)

    @classmethod
    def path_variable_help_message(cls):
        help_message = (
            "Please set it to the absolute path of your git repository folder, e.g.: \n\n"
            "if your repository is at /path/to/your/git/repo, do "
            "export HUGGINGFACE_DIR_PATH=/path/to/your/git\n\n"
            "You can do this by adding the above command to your shell configuration file, "
            "such as .bashrc or .zshrc, or by running it in your terminal before running the script."
        )
        return help_message

    def assert_path_variable_is_set(self, repo_path):
        if repo_path is None:
            raise ValueError(
                f"The HUGGINGFACE_REPO_PATH environment variable is not set. "
                + self.path_variable_help_message()
            )

    def assert_path_variable_is_valid(self, repo_path):
        if not os.path.exists(repo_path):
            raise ValueError(
                f"The HUGGINGFACE_DIR_PATH={repo_path} environment variable is set to a non-existent directory. "
                + self.path_variable_help_message()
            )

    def dir_pathname(self):
        return os.path.join(self.base_local_dir, self.dataset_folder(), 'data')

    def pathname(self):
        return os.path.join(self.dir_pathname(), self.filename())

    def filename(self):
        return f"{self.value()}_{self.year}-{self.month}"

    def url(self):
        return f"https://files.pushshift.io/reddit/{self.dataset_url()}/{self.filename()}.zst"

    def value(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def dataset_url(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def dataset_folder(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def data_processor(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def dataset_schema(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def text_features(self):
        raise NotImplementedError("This method should be implemented in a subclass.")


class CommentsDataset(PushshiftDataset):
    def value(self):
        return 'RC'

    def dataset_folder(self):
        return 'pushshift-reddit-comments'

    def dataset_url(self):
        return 'comments'

    def dataset_schema(self):
        return {
            'author': 'string',
            'body': 'string',
            'controversiality': 'int64',
            'created_utc': 'int64',
            'id': 'string',
            'link_id': 'string',
            'score': 'int64',
            'subreddit': 'string',
            'subreddit_id': 'string'
        }

    def text_features(self):
        return ["body"]


class SubmissionsDataset(PushshiftDataset):
    def value(self):
        return 'RS'

    def dataset_folder(self):
        return 'pushshift-reddit'

    def dataset_url(self):
        return 'submissions'

    def dataset_schema(self):
        return {
            'author': 'string',
            'created_utc': 'int64',
            'id': 'string',
            'num_comments': 'int64',
            'score': 'int64',
            'selftext': 'string',
            'subreddit': 'string',
            'subreddit_id': 'string',
            'title': 'string'
        }

    def text_features(self):
        return ["selftext", "title"]


def month_to_str(month):
    return str(month).zfill(2)
