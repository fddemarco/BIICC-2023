import click

from fasttext_experiment import FasttextExperiment
from posts_type import Submissions, Comments


@click.command()
@click.option("-t", "--posts-type", "posts_type",
              help="Specify the type of Reddit posts to process: s (submissions) / c (comments).",
              type=click.Choice(['submissions', 'comments']),
              prompt=True,
              )
@click.option("-y", "--year", "year",
              help="Specify the year of the posts to process (2012-2018).",
              type=click.IntRange(2012, 2018),
              prompt=True,
              )
@click.option("-wd", "--working-dir", "working_dir",
              help="Specify the working directory.",
              type=click.Path(exists=True, readable=True, writable=True),
              default='.'
              )
@click.option("-e", "--execute-command", "command",
              help="Execute a specified command.",
              type=click.Choice(['texts', 'truncate', 'embeddings', 'compare']),
              default='~/Downloads/fastText-0.9.2/fasttext'
              )
@click.option("-o", "--results-output", "results_dir",
              help="Results folder name",
              default='results'
              )
def app(posts_type: str,
        year: int,
        working_dir: str,
        command: str,
        results_dir: str
        ) -> None:
    """
    Process a specific Reddit POSTS TYPE (submissions or comments) for a given YEAR.
    \f
    :param posts_type: Reddit posts type to process (comments/submissions)
    :param year: Year of the posts (a value between 2012 and 2018)
    :param working_dir: Working directory.
    :param command: Command to process.
    :param results_dir: Results folder name.
    :return: None
    """
    if posts_type == "submissions":
        post_type = Submissions()
    else:
        post_type = Comments()

    experiment = FasttextExperiment(
        year,
        working_dir,
        results_dir,
        post_type
    )
    if command == 'texts':
        if results_dir == 'truncated':
            experiment.generate_texts(pathname=experiment.truncated_data_pathname)
        else:
            experiment.generate_texts()
    elif command == 'truncate':
        experiment.generate_truncated_texts()
    elif command == 'embeddings':
        experiment.save_embeddings_to_csv()
    elif command == 'compare':
        experiment.compare_rankings()
    else:
        raise ValueError("Invalid command.")


if __name__ == "__main__":
    app()
