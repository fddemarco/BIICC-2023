import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow as pa
import pyarrow.dataset as ds

from experiments.reddit_posts import RedditPosts
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
        self.assertEqual(
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

        self.assertEqual(
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

        self.assertEqual(
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

        self.assertEqual(['fun'], subreddits)

    def test_get_most_popular_subreddits_ascending(self):
        df = pd.DataFrame(
            {
                'title': ['1', '2', '3', '4', '5', '6'],
                'selftext': ['', '', '', '', '', ''],
                'subreddit': ['fun', 'fun', 'fun', 'reddit', 'reddit', 'news']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = posts.get_most_popular_subreddits(2).index.tolist()

        self.assertEqual(['reddit', 'fun'], subreddits)

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
        self.assertEqual((2, 300), df.shape)
        subreddits = set(df.index)
        self.assertEqual({'democrats', 'Conservative'}, subreddits)

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
        self.assertEqual({democrats_text, conservative_text}, model.text)

    def test_ranked_subreddits(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'Conservative title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'Conservative body 1'],
                'subreddit': ['democrats', 'democrats', 'Conservative']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = set(posts.get_ranked_subreddits())
        self.assertEqual({'democrats', 'Conservative'}, subreddits)

    def test_not_ranked_subreddits(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'fun title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'fun body 1'],
                'subreddit': ['democrats', 'democrats', 'fun']
            })
        posts, _ = self.create_posts_instance(df)
        subreddits = set(posts.get_ranked_subreddits())
        self.assertEqual({'democrats'}, subreddits)

    def test_truncate_dataset(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'democrats title 2', 'fun title 1'],
                'selftext': ['democrats body 1', 'democrats body 2', 'fun body 1'],
                'subreddit': ['democrats', 'democrats', 'fun'],
                'score': [10, 5, 1]
            })
        expected = pd.DataFrame(
            {
                'title': ['fun title 1', 'democrats title 1'],
                'selftext': ['fun body 1', 'democrats body 1'],
                'subreddit': ['fun', 'democrats'],
                'score': [1, 10],
                'text': ['fun title 1 fun body 1', 'democrats title 1 democrats body 1']
            })
        posts, _ = self.create_posts_instance(df)
        df = posts.truncate_dataset(35)
        self.assertTrue(
            expected.equals(df)
        )

    def test_truncate_dataset_first_post_exceeds_threshold(self):
        df = pd.DataFrame(
            {
                'title': ['democrats title 1', 'title 2', 'fun title 1'],
                'selftext': ['', '', ''],
                'subreddit': ['democrats', 'democrats', 'fun'],
                'score': [10, 5, 1]
            })
        expected = pd.DataFrame(
            {
                'title': ['fun title 1', 'title 2'],
                'selftext': ['', ''],
                'subreddit': ['fun', 'democrats'],
                'score': [1, 5],
                'text': ['fun title 1', 'title 2']
            })
        posts, _ = self.create_posts_instance(df)
        df = posts.truncate_dataset(15)
        assert_frame_equal(expected, df)


if __name__ == '__main__':
    unittest.main()
