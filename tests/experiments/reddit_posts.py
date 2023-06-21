import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.experiments.reddit_posts import RedditPosts
from experiment_stub import FasttextExperimentStub
from model_stub import ModelStub


class RedditPostsTestCase(unittest.TestCase):
    def create_posts_instance(self, df):
        data = ds.dataset(pa.Table.from_pandas(df).to_batches())
        text_sink = FasttextExperimentStub()
        posts = RedditPosts.from_submissions(data, text_sink)
        return posts, text_sink

    def test_generate_text(self):
        df = pd.DataFrame(
            {
                'title': ['fun title 1', 'fun title 2', 'reddit title 1'],
                'selftext': ['fun body 1', 'fun body 2', 'reddit body 1'],
                'subreddit': ['fun', 'fun', 'reddit']
            })
        posts, text_sink = self.create_posts_instance(df)
        posts.generate_text()
        self.assertEquals(
            {'reddit title 1 reddit body 1',
             'fun title 1 fun body 1 fun title 2 fun body 2'}, text_sink.text)

    def test_generate_text_empty_title(self):
        df = pd.DataFrame(
            {
                'title': ['', 'fun title 2', 'reddit title 1'],
                'selftext': ['fun body 1', 'fun body 2', 'reddit body 1'],
                'subreddit': ['fun', 'fun', 'reddit']
             })
        posts, text_sink = self.create_posts_instance(df)
        posts.generate_text()

        self.assertEquals(
            {'reddit title 1 reddit body 1',
             'fun body 1 fun title 2 fun body 2'}, text_sink.text)

    def test_generate_text_empty_selftext(self):
        df = pd.DataFrame(
            {
                'title': ['fun title 1', 'fun title 2', 'reddit title 1'],
                'selftext': ['', 'fun body 2', 'reddit body 1'],
                'subreddit': ['fun', 'fun', 'reddit']
             })
        posts, text_sink = self.create_posts_instance(df)
        posts.generate_text()

        self.assertEquals(
            {'reddit title 1 reddit body 1',
             'fun title 1 fun title 2 fun body 2'}, text_sink.text)

    def test_get_most_popular_subreddits_up_to_k(self):
        df = pd.DataFrame(
            {
                'title': ['1', '2', '3'],
                'selftext': ['', '', ''],
                'subreddit': ['fun', 'fun', 'reddit']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = posts.get_most_popular_subreddits(1).index.tolist()

        self.assertEquals(['fun'], subreddits)

    def test_get_most_popular_subreddits_ascending(self):
        df = pd.DataFrame(
            {
                'title': ['1', '2', '3', '4', '5', '6'],
                'selftext': ['', '', '', '', '', ''],
                'subreddit': ['fun', 'fun', 'fun', 'reddit', 'reddit', 'news']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = posts.get_most_popular_subreddits(2).index.tolist()

        self.assertEquals(['reddit', 'fun'], subreddits)

    def test_generate_embeddings(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'Conservative title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'Conservative body 1'],
                'subreddit': ['democrats', 'democrats', 'Conservative']
            })
        posts, _ = self.create_posts_instance(df)
        model = ModelStub()
        df = posts.generate_embeddings_for(model)
        self.assertEquals((2, 300), df.shape)
        subreddits = set(df.index)
        self.assertEquals({'democrats', 'Conservative'}, subreddits)

    def test_text_generate_embeddings(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'Conservative title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'Conservative body 1'],
                'subreddit': ['democrats', 'democrats', 'Conservative']
            })
        posts, _ = self.create_posts_instance(df)
        model = ModelStub()
        posts.generate_embeddings_for(model)
        democrats_text = 'democrats title 1 democrats body 1 democrats title 2 democrats body 2'
        conservative_text = 'Conservative title 1 Conservative body 1'
        self.assertEquals({democrats_text, conservative_text}, model.text)

    def test_ranked_subreddits(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'Conservative title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'Conservative body 1'],
                'subreddit': ['democrats', 'democrats', 'Conservative']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = set(posts.get_ranked_subreddits())
        self.assertEquals({'democrats', 'Conservative'}, subreddits)

    def test_not_ranked_subreddits(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'fun title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'fun body 1'],
                'subreddit': ['democrats', 'democrats', 'fun']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = set(posts.get_ranked_subreddits())
        self.assertEquals({'democrats'}, subreddits)


if __name__ == '__main__':
    unittest.main()
