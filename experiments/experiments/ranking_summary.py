import pandas as pd


class InvalidMetrics(ValueError):
    """
        Metrics should include year of dataset and model used to generate the metrics
    """
    pass


class RankingSummarizer:
    def __init__(self, results, model=None, year=None):
        self.validate_input(model, results, year)
        data = pd.DataFrame(results)
        self.data = self.normalize_data(data)

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

    def normalize_data(self, data):
        data = data.sort_values([self.model_field, self.year_field])
        return data.reset_index(drop=True)

    def union(self, summary):
        return RankingSummarizer(summary.concat(self.data))

    def concat(self, data):
        concat_data = pd.concat([self.data, data])
        return self.normalize_data(concat_data)

    def to_pandas(self):
        return self.data

    def __eq__(self, other):
        return other.equals_to(self.data)

    def equals_to(self, data):
        return self.data.equals(data)
