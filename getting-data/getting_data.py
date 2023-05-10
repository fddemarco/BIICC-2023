from huggingface_hub import snapshot_download


class GetData:
    def __init__(self, year, local_dir=''):
        self.year = year
        self.local_dir = local_dir

    def download_data(self):
        snapshot_download(
            repo_id="fddemarco/pushshift-reddit",
            repo_type="dataset",
            allow_patterns=self.dataset_pattern(),
            local_dir=self.local_dir,
            local_dir_use_symlinks=False)

    def huggingface_dataset(self):
        raise NotImplementedError("Should be implemented in a subclass")

    def dataset_pattern(self):
        raise NotImplementedError("Should be implemented in a subclass")


class GetSubmissions(GetData):
    def huggingface_dataset(self):
        return "pushshift-reddit"

    def dataset_pattern(self):
        return f"data/{self.dataset}_{self.year}-[0-9]*_[0-9]*.parquet"

    @property
    def dataset(self):
        return "RS"


class GetComments(GetData):
    def huggingface_dataset(self):
        return "pushshift-reddit-comments"

    def dataset_pattern(self):
        return f"data/{self.dataset}_{self.year}-[0-9]*.parquet"

    @property
    def dataset(self):
        return "RC"
