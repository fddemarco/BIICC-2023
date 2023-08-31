"""
    Reddit post types
"""


class PostsType:
    """
    Abstract class for posts type
    """

    def texts_from(self, reddit_posts, data):
        """
        Double-dispatch for getting text from data
        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def posts_type(self, experiment):
        """
        Double-dispatch for getting corresponding dataset folder
        name based on posts type
        """
        raise NotImplementedError("Should be implemented in a subclass.")


class Submissions(PostsType):
    """
    Submissions Posts type
    """

    def texts_from(self, reddit_posts, data):
        return reddit_posts.submissions_text(data)

    def posts_type(self, experiment):
        return experiment.submissions_dataset


class Comments(PostsType):
    """
    Comments Posts type
    """

    def texts_from(self, reddit_posts, data):
        return reddit_posts.comments_text(data)

    def posts_type(self, experiment):
        return experiment.comments_dataset
