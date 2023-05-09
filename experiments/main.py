from fasttext_experiment import FasttextExperimentForSubmissions
from fasttext_experiment import FasttextExperimentForComments

if __name__ == "__main__":
    year = input("Enter year to process (2012-2018): ")
    texts_dir = input("Enter pathname to texts dir: ")
    data_dir = input("Enter pathname to data dir: ")
    results_dir = input("Enter pathname to results dir: ")
    fasttext_pathname = input("Enter pathname to fasttext executable: ")
    dataset = input("Enter dataset (RS or RC): ")

    if dataset == "RS":
        experiment = FasttextExperimentForSubmissions(year, data_dir, texts_dir, results_dir, fasttext_pathname)
    else:
        experiment = FasttextExperimentForComments(year, data_dir, texts_dir, results_dir, fasttext_pathname)
    experiment.run_experiment()
