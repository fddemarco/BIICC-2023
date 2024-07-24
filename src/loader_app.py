import click

from dimensions.loader.dataset import Submissions
from dimensions.loader.dataset import Comments


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
@click.option("--download",
              help="Download specified dataset.",
              is_flag=True,
              show_default=True,
              default=False
              )
@click.option("--clean",
              help="Clean specified dataset.",
              is_flag=True,
              show_default=True,
              default=False
              )
def app(posts_type: str,
        year: int,
        working_dir: str,
        download: bool,
        clean: bool
        ) -> None:
    """
    Process a specific Reddit POSTS TYPE (submissions or comments) for a given YEAR.
    \f
    :param posts_type: Reddit posts type to process (comments/submissions)
    :param year: Year of the posts (a value between 2012 and 2018)
    :param working_dir: Working directory.
    :param download: Download specified dataset.
    :param clean: Clean downloaded dataset.
    :return: None
    """
    dataset = Submissions(year, working_dir) if posts_type == 'submissions' else Comments(year, working_dir)

    if download:
        dataset.download_data()
    if clean:
        dataset.clean_dataset()


if __name__ == "__main__":
    app()
