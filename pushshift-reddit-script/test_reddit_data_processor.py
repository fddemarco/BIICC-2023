import string
import unittest
from reddit_data_processor import clean_text
from reddit_data_processor import SubmissionsProcessor
from reddit_data_processor import CommentsProcessor

PATH = '/home/franco/Desktop/BIICC/data/RC_2006-01.zst'


class RedditDataProcessorTestCase(unittest.TestCase):
    def test_clean_text_manges_None_values(self):
        with self.assertRaisesRegex(TypeError, "Input text cannot be None."):
            clean_text(None)

    def test_clean_text_empty_string_is_clean(self):
        self.assertEqual('', clean_text(''))

    def test_clean_text_comment_removed_by_user(self):  # comment:'[removed]' (by user)
        self.assertEqual('', clean_text('[removed]'))

    def test_clean_text_comment_deleted_by_moderator(self): # comment:'[deleted]' (by moderator)
        self.assertEqual('', clean_text('[deleted]'))

    def test_clean_text_lowers_cases(self):
        self.assertEqual('comment', clean_text('CoMmeNt'))

    def test_clean_text_trims_whitespaces(self):
        self.assertEqual('comment', clean_text(' comment '))

    def test_clean_text_removes_special_whitespaces(self):
        special_whitespaces = string.whitespace.replace(' ', '')
        self.assertEqual('', clean_text(special_whitespaces))

    def test_clean_text_removes_multiple_whitespaces_between_words(self):
        self.assertEqual('comment text', clean_text('comment  text'))

    def test_clean_text_add_single_whitespace_between_punctuation_character_and_words(self):
        for c in string.punctuation:
            self.assertEqual(f'comment {c} text', clean_text(f'comment{c} text'))

    def test_text_keys_for_submissions(self):
        processor = SubmissionsProcessor('')
        text_keys = processor.text_keys()
        self.assertEqual({'title', 'selftext'}, text_keys)

    def test_text_keys_for_comments(self):
        processor = CommentsProcessor('')
        text_keys = processor.text_keys()
        self.assertEqual({'body'}, text_keys)

    def test_selected_features_for_submissions(self):
        processor = SubmissionsProcessor('')
        features = processor.selected_features()
        self.assertEqual({
            'author',
            'created_utc',
            'id',
            'num_comments',
            'score',
            'selftext',
            'subreddit',
            'subreddit_id',
            'title'
            }, features)

    def test_selected_features_for_comments(self):
        processor = CommentsProcessor('')
        features = processor.selected_features()
        self.assertEqual({
            'author',
            'body',
            'controversiality',
            'created_utc',
            'id',
            'link_id',
            'score',
            'subreddit',
            'subreddit_id'
            }, features)

    @staticmethod
    def submission_with_clean_text():
        return {'title': 'title', 'selftext': 'content of submission .'}

    @staticmethod
    def empty_data(processor):
        return {k: None for k in processor.selected_features()}

    @staticmethod
    def submission_with_dirty_data():
        return {'feature': 42, 'title': ' TiTle ', 'selftext': '   content of  suBmission.  '}

    def clean_data(self, processor):
        data = self.empty_data(processor)
        texts = self.submission_with_clean_text()
        for k, v in texts.items():
            data[k] = v
        return data

    def test_clean_texts_manages_missing_text_keys(self):
        processor = SubmissionsProcessor('')
        with self.assertRaisesRegex(TypeError, "Input text cannot be None."):
            processor.process_features({}, processor.text_keys(), clean_text)

    def test_clean_texts_cleans_text(self):
        processor = SubmissionsProcessor('')
        data = {'number': 1}
        processor.process_features(data, ['number'], lambda x: x + 1)
        self.assertEqual({'number': 2}, data)

    def test_process_data(self):
        processor = SubmissionsProcessor('')
        data = self.submission_with_dirty_data()
        data = processor.process_data(data)
        self.assertEqual(data, self.clean_data(processor))

    def test_empty_data_has_no_text(self):
        processor = SubmissionsProcessor('')
        self.assertFalse(processor.has_text({}))

    def test_data_without_text_keys_has_no_text(self):
        processor = SubmissionsProcessor('')
        self.assertFalse(processor.has_text({'feature': 'text'}))

    def test_data_with_no_text_has_no_text(self):
        processor = SubmissionsProcessor('')
        self.assertFalse(processor.has_text({'title': ''}))

    def test_data_with_text_has_text(self):
        processor = SubmissionsProcessor('')
        self.assertTrue(processor.has_text(self.submission_with_clean_text()))


if __name__ == '__main__':
    unittest.main()
