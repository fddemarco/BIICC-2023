from enum import Enum
from experiments.posts_type import Submissions, Comments


class Options(str, Enum):
    """
    Options class type for CLI argument
    """
    @classmethod
    def help_message(cls: type) -> str:
        raise NotImplementedError('Should be implemented in a subclass')

    def __str__(self):
        return self.value


class Types(Options):
    """
    Reddit Posts type for CLI argument
    """
    SUBMISSIONS = 'submissions'
    COMMENTS = 'comments'

    @classmethod
    def help_message(cls: type) -> str:
        return 'Reddit posts type to process'

    def to_model(self):
        if self.value == self.SUBMISSIONS:
            return Submissions()
        else:
            return Comments()


class Dataset(Options):
    """
    Posts Dataset type for CLI argument
    """
    ORIGINAL = 'original'
    TRUNCATED = 'truncated'

    @classmethod
    def help_message(cls: type) -> str:
        return 'Dataset folder name'


class ResultDir(Options):
    """
    Results folder name for CLI argument
    """
    RESULTS = 'results'
    PRETRAINED = 'pretrained'

    @classmethod
    def help_message(cls: type) -> str:
        return 'Results folder name'


class Command(Options):
    """
    Command type for CLI argument
    """
    TEXTS = 'texts'
    TRUNCATE = 'truncate'
    EMBEDDINGS = 'embeddings'
    COMPARE = 'compare'

    @classmethod
    def help_message(cls: type) -> str:
        return 'Command to process'

    def apply(self, experiment):
        if self == Command.TEXTS:
            experiment.apply_texts()
        elif self == Command.TRUNCATE:
            experiment.apply_truncate()
        elif self == Command.EMBEDDINGS:
            experiment.apply_embeddings()
        elif self == Command.COMPARE:
            experiment.apply_compare()
