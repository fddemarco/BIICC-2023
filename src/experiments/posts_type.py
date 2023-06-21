class PostsType:
    def texts_from(self, reddit_posts, df):
        raise NotImplementedError('Should be implemented in a subclass.')

    def dataset_type(self, experiment):
        raise NotImplementedError('Should be implemented in a subclass.')


class Submissions(PostsType):
    def texts_from(self, reddit_posts, df):
        return reddit_posts.submissions_text(df)

    def dataset_type(self, experiment):
        return experiment.submissions_dataset


class Comments(PostsType):
    def texts_from(self, reddit_posts, df):
        return reddit_posts.comments_text(df)

    def dataset_type(self, experiment):
        return experiment.comments_dataset

