import numpy as np


class ModelStub:
    def __init__(self):
        self.text = set()

    def get_sentence_vector(self, text):
        self.text.add(text)
        return self

    def astype(self, _type):
        return np.array([0] * 300)
