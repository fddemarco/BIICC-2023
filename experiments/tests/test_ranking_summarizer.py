import unittest

import pytest
import pandas as pd
import pandas.testing as pdt

from experiments.ranking import Ranking
from experiments.ranking_summarizer import RankingSummarizer, InvalidMetrics


def ranking_metrics():
    return Ranking(
        {
            'democrats': -10,
            'EnoughLibertarianSpam': -9,
            'hillaryclinton': -8,
            'progressive': -7,
            'BlueMidterm2018': -6,
            'EnoughHillHate': -5,
            'Enough_Sanders_Spam': -4,
            'badwomensanatomy': -3,
            'racism': -2,
            'GunsAreCool': 1,
            'Christians': -1,
            'The_Farage': 2,
            'new_right': 3,
            'conservatives': 4,
            'metacanada': 5,
            'Mr_Trump': 6,
            'NoFapChristians': 7,
            'TrueChristian': 8,
            'The_Donald': 9,
            'Conservative': 10
        }).evaluate_ranking_metrics()


class RankingTestCase(unittest.TestCase):
    def test_00(self):
        metrics = ranking_metrics()
        summarizer = RankingSummarizer(metrics, '10k', 2012)
        self.assertEqual(2012, summarizer.to_pandas().year[0])
        self.assertEqual('10k', summarizer.to_pandas().model[0])
        self.assertEqual(metrics['AUC ROC'][0], summarizer.to_pandas()['AUC ROC'][0])

    def test_01(self):
        metrics = ranking_metrics()
        summarizer1 = RankingSummarizer(metrics, '10k', 2012)
        summarizer2 = RankingSummarizer(metrics, '10k', 2013)
        concat_data = summarizer1.union(summarizer2).to_pandas()
        pdt.assert_series_equal(concat_data.year, pd.Series([2012, 2013], name='year'))

    def test_02(self):
        metrics = ranking_metrics()
        summarizer1 = RankingSummarizer(metrics, '10k', 2013)
        summarizer2 = RankingSummarizer(metrics, '10k', 2012)
        concat_data = summarizer1.union(summarizer2).to_pandas()
        pdt.assert_series_equal(concat_data.year, pd.Series([2012, 2013], name='year'))

    def test_03(self):
        metrics = ranking_metrics()
        summarizer1 = RankingSummarizer(metrics, 'b', 2012)
        summarizer2 = RankingSummarizer(metrics, 'a', 2013)
        concat_data = summarizer1.union(summarizer2).to_pandas()
        pdt.assert_series_equal(concat_data.year, pd.Series([2013, 2012], name='year'))
        pdt.assert_series_equal(concat_data.model, pd.Series(['a', 'b'], name='model'))

    def test_04(self):
        metrics = ranking_metrics()
        s1 = RankingSummarizer(metrics, 'b', 2012)
        s2 = RankingSummarizer(metrics, 'a', 2013)
        s3 = RankingSummarizer(metrics, 'a', 2012)
        s4 = s1.union(s2)
        concat_data = s4.union(s3).to_pandas()
        pdt.assert_series_equal(concat_data.year, pd.Series([2012, 2013, 2012], name='year'))
        pdt.assert_series_equal(concat_data.model, pd.Series(['a', 'a', 'b'], name='model'))

    def test_05(self):
        metrics = ranking_metrics()
        with pytest.raises(InvalidMetrics):
            RankingSummarizer(metrics)
