from getting_data import GetSubmissions
from getting_data import GetComments

dataset = input("Enter dataset (RS/RC): ")
year = input("Enter year (2012-2018): ")

if dataset == "RS":
    GetSubmissions(year).download_data()
elif dataset == "RC":
    GetComments(year).download_data()
else:
    raise ValueError("Invalid dataset: dataset should be RS or RC")

