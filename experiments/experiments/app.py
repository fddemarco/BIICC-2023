"""
CLI is defined in this module.
"""

from enum import Enum
from pathlib import Path

from typing_extensions import Annotated
import typer


from experiments.fasttext_experiment import FasttextExperiment
from experiments.posts_type import Submissions, Comments


class Options(str, Enum):
    """
    Options class type for CLI argument
    """
    def __str__(self):
        return self.value


class Types(Options):
    """
    Reddit Posts type
    """
    SUBMISSIONS = "submissions"
    COMMENTS = "comments"


class Dataset(Options):
    """
    Posts Dataset used
    """
    ORIGINAL = "original"
    TRUNCATED = "truncated"


class OutputDir(Options):
    """
    Output directory
    """
    RESULTS = "results"
    PRETRAINED = "pretrained"


class Command(Options):
    """
    App command
    """
    TEXTS = 'texts'
    TRUNCATE = 'truncate'
    EMBEDDINGS = 'embeddings'
    COMPARE = 'compare'


app = typer.Typer()


@app.command()
def main(working_dir: Annotated[Path,
         typer.Argument(help='Working directory.')],
         command: Annotated[Command,
         typer.Argument(help='Command to process.')],
         posts_type: Annotated[Types,
         typer.Argument(
             help='Reddit posts type to process (comments/submissions)')] = Types.SUBMISSIONS,
         _from: Annotated[int,
         typer.Argument(help='Start year of the posts (a value between 2012 and 2018)',
                        min=2012,
                        max=2018)] = 2012,
         _to: Annotated[int,
         typer.Argument(help='End year of the posts (a value between 2012 and 2018)',
                        min=2012,
                        max=2018)] = 2018,
         results_dir: Annotated[OutputDir,
         typer.Argument(help='Results folder name.')] = OutputDir.RESULTS,
         dataset: Annotated[Dataset,
         typer.Argument(help='Dataset folder name.')] = Dataset.ORIGINAL,
         ):
    """
    Process a specific Reddit POSTS TYPE (submissions or comments) for a give range of YEARs.
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
        if command == Command.TEXTS:
            experiment.generate_texts()
        elif command == Command.TRUNCATE:
            experiment.generate_truncated_texts()
        elif command == Command.EMBEDDINGS:
            experiment.save_embeddings_to_csv()
        elif command == Command.COMPARE:
            experiment.compare_rankings()
        else:
            raise ValueError("Invalid command.")
