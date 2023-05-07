from tqdm import tqdm
from pushshift_dataset import SubmissionsDataset
from pushshift_dataset import CommentsDataset
from file_handler import FileHandler


def process_monthly_dataset(dataset_cls, month, year, process):
    file_handler = FileHandler(dataset_cls(month, year))
    process(file_handler)


def run(dataset, years, months, process):
    for year in tqdm(years):
        for month in tqdm(months, leave=False):
            print(f"\n Processing month {month} of year {year}...")
            process_monthly_dataset(dataset, month, year, process)


def user_prompt(s):
    return f"Enter the start and end {s} (separated by space): "
    

def read_numerical_input(s):
    return map(int, input(s).split(' '))


def read_dataset_flag():
    dataset_flag = input("Enter the dataset to process (-c: comments/ -s: submissions): ")
    if dataset_flag == "-c":
        return CommentsDataset
    elif dataset_flag == "-s":
        return SubmissionsDataset
    else:
        raise ValueError("Invalid Dataset")


def processing_prompt(processing_commands, handler):
    if "download" in processing_commands:
        handler.download()
    if "process" in processing_commands:
        handler.process()
    if "delete" in processing_commands:
        handler.delete_compressed()
    

if __name__ == '__main__':
   
    start_year, end_year = read_numerical_input(user_prompt("years"))
    start_month, end_month = read_numerical_input(user_prompt("months"))
    dataset = read_dataset_flag()
    commands = input("Enter the commands to run, separated by spaces  (download/process/delete): ").split(' ')
    run(
        dataset,
        range(start_year, end_year+1),
        range(start_month, end_month+1),
        lambda handler: processing_prompt(commands, handler)
        )
