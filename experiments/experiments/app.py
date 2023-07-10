from enum import Enum

import typer
from typing_extensions import Annotated
from pathlib import Path

from experiments.fasttext_experiment import FasttextExperiment
from experiments.posts_type import Submissions, Comments


class Options(str, Enum):
    def __str__(self):
        return self.value


class Types(Options):
    submissions = "submissions"
    comments = "comments"


class Dataset(Options):
    original = "original"
    truncated = "truncated"


class OutputDir(Options):
    results = "results"
    pretrained = "pretrained"


class Command(Options):
    texts = 'texts'
    truncate = 'truncate'
    embeddings = 'embeddings'
    compare = 'compare'


app = typer.Typer()


@app.command()
def main(working_dir: Annotated[Path,
         typer.Argument(help='Working directory.')],
         command: Annotated[Command,
         typer.Argument(help='Command to process.')],
         posts_type: Annotated[Types,
         typer.Argument(help='Reddit posts type to process (comments/submissions)')] = Types.submissions,
         _from: Annotated[int,
         typer.Argument(help='Start year of the posts (a value between 2012 and 2018)',
                        min=2012,
                        max=2018)] = 2012,
         _to: Annotated[int,
         typer.Argument(help='End year of the posts (a value between 2012 and 2018)',
                        min=2012,
                        max=2018)] = 2018,
         results_dir: Annotated[OutputDir,
         typer.Argument(help='Results folder name.')] = OutputDir.results,
         dataset: Annotated[Dataset,
         typer.Argument(help='Dataset folder name.')] = Dataset.original,
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
