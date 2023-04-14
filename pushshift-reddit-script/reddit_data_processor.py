import orjson
import os
import re


class RedditDataProcessor:
    def __init__(self, file):
        self.file_name = file

    def reduce_data(self):
        output_file = self.small_format()
        if not os.path.exists(output_file) and os.path.exists(self.file_name):
            self.select_features(output_file)

    def select_features(self, output_file):
        with open(self.file_name, "r") as r:
            with open(output_file, "wb") as w:
                for json_obj in r:
                    raw_data = orjson.loads(json_obj)
                    data = self.process_json(raw_data)
                    if self.has_text(data):
                        w.write(orjson.dumps(data) + bytes('\n', 'utf-8'))

    def text_keys(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def selected_features(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def has_text(self, data):
        return any(data[k] for k in self.text_keys())

    def process_json(self, json_data):
        features = self.selected_features()
        selected_data = {f: json_data.get(f, None) for f in features}
        self.clean_texts(selected_data)
        return selected_data

    def clean_texts(self, selected_data):
        for k in self.text_keys():
            text = selected_data[k]
            selected_data[k] = clean_text('' if text is None else text)

    def small_format(self):
        return f"{self.file_name}_small.json"


class CommentsProcessor(RedditDataProcessor):
    def text_keys(self):
        return ["body"]

    def selected_features(self):
        return ['author',
                'body',
                'controversiality',
                'created_utc',
                'id',
                'link_id',
                'score',
                'subreddit',
                'subreddit_id'
                ]


class SubmissionsProcessor(RedditDataProcessor):
    def text_keys(self):
        return ["selftext", "title"]

    def selected_features(self):
        return ['author',
                'created_utc',
                'id',
                'num_comments',
                'score',
                'selftext',
                'subreddit',
                'subreddit_id',
                'title'
                ]


# r/subreddit to mention subreddit
# u/username to mention user
def clean_text(text):
    text = text.replace('[removed]', '')  # comment:'[removed]' (by user)
    text = text.replace('[deleted]', '')  # comment:'[deleted]' (by moderator)
    if text:
        text = re.sub(r"([.!?,'/()])", r" \1", text)  # Matches punctuation characters and captures them
        text = re.sub(r"\s+", ' ', text)  # Matches Unicode whitespace characters
        text = text.lower()
    return text
