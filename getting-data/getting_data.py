from huggingface_hub import snapshot_download
import pathlib


class RedditPostsDataset:
    def __init__(self, year, working_dir):
        self.year = str(year)
        self.working_dir = pathlib.Path(working_dir)

    def download_data(self):
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            allow_patterns=self.hf_filename_pattern,
            local_dir=self.hf_dataset_dir,
            cache_dir=self.cache_dir,
            local_dir_use_symlinks=True)

    @property
    def repo_id(self):
        return f"fddemarco/{self.huggingface_dataset}"

    def wd_subdir(self, folder):
        return self.working_dir / folder

    @property
    def data_dir(self):
        return self.wd_subdir('data')

    @property
    def cache_dir(self):
        return self.wd_subdir('cache')

    @property
    def hf_dataset_dir(self):
        return self.data_dir / self.huggingface_dataset / self.year

    @property
    def huggingface_dataset(self):
        raise NotImplementedError("Should be implemented in a subclass")

    @property
    def hf_filename_pattern(self):
        return f"data/{self.dataset}_{self.year}-[0-9]*{self.split_pattern}.parquet"

    @property
    def split_pattern(self):
        raise NotImplementedError("Should be implemented in a subclass")

    @property
    def dataset(self):
        raise NotImplementedError("Should be implemented in a subclass")


class Submissions(RedditPostsDataset):
    @property
    def huggingface_dataset(self):
        return "pushshift-reddit"

    @property
    def split_pattern(self):
        return '_[0-9]*'

    @property
    def dataset(self):
        return "RS"


class Comments(RedditPostsDataset):
    @property
    def huggingface_dataset(self):
        return "pushshift-reddit-comments"

    @property
    def split_pattern(self):
        return ''

    @property
    def dataset(self):
        return "RC"
