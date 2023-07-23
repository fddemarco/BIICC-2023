import pandas as pd


class InvalidMetrics(ValueError):
    """
        Metrics should include year of dataset and model used to generate the metrics
    """
    pass


class RankingSummarizer:
    def __init__(self, data, model=None, year=None):
        self.validate_input(model, data, year)
        self.data = pd.DataFrame(data)

    def validate_input(self, model, results, year):
        if model is None or year is None:
            if (
                self.model_field not in results or
                self.year_field not in results
            ):
                raise InvalidMetrics
        else:
            results[self.model_field] = [model]
            results[self.year_field] = [year]

    @property
    def year_field(self):
        return 'year'

    @property
    def model_field(self):
        return 'model'

    def normalize_data(self):
        data = self.data.sort_values([self.model_field, self.year_field])
        return data.reset_index(drop=True).round(2)

    def union(self, summary):
        return RankingSummarizer(summary.concat(self.data))

    def concat(self, data):
        return pd.concat([self.data, data])

    def to_pandas(self):
        return self.normalize_data()

    def __eq__(self, other):
        return other.equals_to(self.data)

    def equals_to(self, data):
        return self.data.equals(data)
