from getting_data import GetSubmissions
from getting_data import GetComments

dataset = input("Enter dataset (RS/RC): ")
year = input("Enter year (2012-2018): ")
local_dir = input("Enter dir of symlink: ")
cache_dir = input("Enter dir of dataset: ")

if dataset == "RS":
    downloader_cls = GetSubmissions
elif dataset == "RC":
    downloader_cls = GetComments
else:
    raise ValueError("Invalid dataset: dataset should be RS or RC")

downloader = downloader_cls(year, local_dir, cache_dir)
downloader.download_data()
