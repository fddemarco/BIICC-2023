import click
from experiments.fasttext_experiment import FasttextExperiment
from experiments.posts_type import Submissions, Comments


@click.command()
@click.option("-t", "--posts-type", "posts_type",
              help="Specify the type of Reddit posts to process: submissions / comments",
              type=click.Choice(['submissions', 'comments']),
              default='submissions'
              )
@click.option("--from", "_from",
              help="Specify the start year of the posts to process (2012-2018)",
              type=click.IntRange(2012, 2018),
              default=2012
              )
@click.option("--to", "_to",
              help="Specify the end year of the posts to process (2012-2018)",
              type=click.IntRange(2012, 2018),
              default=2018
              )
@click.option("-wd", "--working-dir", "working_dir",
              help="Specify the working directory",
              type=click.Path(exists=True, readable=True, writable=True),
              default='.'
              )
@click.option("-e", "--execute-command", "command",
              help="Execute a specified command",
              type=click.Choice(['texts', 'truncate', 'embeddings', 'compare']),
              default='~/Downloads/fastText-0.9.2/fasttext'
              )
@click.option("-o", "--results-output", "results_dir",
              help="Results folder name",
              default='results'
              )
@click.option("-d", "--dataset", "dataset",
              help="Dataset folder name",
              type=click.Choice(['original', 'truncated']),
              default='original'
              )
def app(posts_type: str,
        _from: int,
        _to: int,
        working_dir: str,
        command: str,
        results_dir: str,
        dataset: str,
        ) -> None:
    """
    Process a specific Reddit POSTS TYPE (submissions or comments) for a given YEAR.
    \f
    :param posts_type: Reddit posts type to process (comments/submissions)
    :param _from: Start year of the posts (a value between 2012 and 2018)
    :param _to: End year of the posts (a value between 2012 and 2018)
    :param working_dir: Working directory.
    :param command: Command to process.
    :param results_dir: Results folder name.
    :param dataset: Dataset folder name.
    :return: None
    """
    if posts_type == "submissions":
        post_type = Submissions()
    else:
        post_type = Comments()

    for year in range(_from, _to+1):
        experiment = FasttextExperiment(
            year,
            working_dir,
            results_dir,
            post_type,
            dataset
        )
        if command == 'texts':
            experiment.generate_texts()
        elif command == 'truncate':
            experiment.generate_truncated_texts()
        elif command == 'embeddings':
            experiment.save_embeddings_to_csv()
        elif command == 'compare':
            experiment.compare_rankings()
        else:
            raise ValueError("Invalid command.")
