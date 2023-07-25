"""
CLI is defined in this module.
"""

from pathlib import Path

from typing_extensions import Annotated
import typer


from experiments.fasttext_experiment import Experiment
from experiments.options import Command, Types, ResultDir, Dataset


def range_help():
    """Help message for 'from' and 'to' arguments"""
    return 'Year between 2012 and 2018'


app = typer.Typer()


@app.command()
def main(working_dir: Annotated[Path,
                                typer.Argument(help='Working directory')],
         command: Annotated[Command,
                            typer.Argument(help=Command.help_message())],
         posts_type: Annotated[Types,
                               typer.Argument(
                                 help=Types.help_message())] = Types.SUBMISSIONS,
         _from: Annotated[int,
                          typer.Argument(help=range_help(),
                                         min=2012,
                                         max=2018)] = 2012,
         _to: Annotated[int,
                        typer.Argument(help=range_help(),
                                       min=2012,
                                       max=2018)] = 2018,
         results_dir: Annotated[ResultDir,
                                typer.Argument(help=ResultDir.help_message())] = ResultDir.RESULTS,
         dataset: Annotated[Dataset,
                            typer.Argument(help=Dataset.help_message())] = Dataset.ORIGINAL,
         ):
    """
    Apply a COMMAND over a Reddit POSTS TYPE DATASET for a given range of YEARs.
    """

    for year in range(_from, _to+1):
        experiment = Experiment(
            year,
            working_dir,
            results_dir,
            posts_type.to_model(),
            dataset
        )
        command.apply(experiment)
