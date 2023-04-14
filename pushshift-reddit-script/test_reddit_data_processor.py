import string
import unittest
from reddit_data_processor import clean_text


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


if __name__ == '__main__':
    unittest.main()
