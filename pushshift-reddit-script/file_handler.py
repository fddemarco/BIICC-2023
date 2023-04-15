import os
from glob import glob
import pandas as pd


class FileHandler:
    def __init__(self, file):
        change_wd(file)
        self.pathname = file.pushshift_path()
        self.url = file.url()
        self.data_processor = file.data_processor()

    def download(self):
        if not self.downloaded():
            os.system(f"wget -O {self.pathname + '.zst'} {self.url}")

    def downloaded(self):
        existing_files = glob(self.pathname + '*')
        return any(existing_files)

    def decompress(self):
        if not self.decompressed():
            compressed_file = self.compressed_format()
            os.system(f'zstd -d --rm --long=31 {compressed_file}')

    def decompressed(self):
        existing_files = glob(self.pathname + self.raw_format())
        return any(existing_files)

    def format_path(self, suffix):
        return self.pathname + suffix

    def compressed_format(self):
        return self.format_path(self.compressed_suffix())

    def raw_format(self):
        return self.pathname

    def small_format(self):
        return self.format_path(self.small_suffix())

    def parquet_format(self):
        return self.format_path('.parquet')

    @staticmethod
    def compressed_suffix():
        return '.zst'

    @staticmethod
    def small_suffix():
        return '_small.json'

    def delete(self, suffix=''):
        file = self.pathname + suffix
        if os.path.exists(file):
            os.remove(file)

    def reduce_data(self):
        self.data_processor.reduce_data()
        self.delete()


def change_wd(file):
    actual_dir = file.build_local_path()
    change_wd_to(actual_dir)


def change_wd_to(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    os.chdir(folder_path)


def to_parquet(filename_pattern):
    splits = glob(filename_pattern)
    for file in splits:
        json = pd.read_json(file, lines=True)
        filename = file.rpartition('_')[0]
        json.to_parquet(filename + '.parquet')
        os.remove(file)


def push_to_hub(filename_pattern):
    files = glob(filename_pattern)
    for file in files:
        os.system(f"git add {file}")
        os.system(f"git commit -m 'Uploaded file: {file}'")
        os.system("git push")
