"""Unit tests for trend_narrative.extractor.InsightExtractor."""

import math

import numpy as np
import pytest

from trend_narrative.extractor import InsightExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stable_series():
    x = np.arange(2010, 2022, dtype=float)
    rng = np.random.default_rng(1)
    y = 100.0 + rng.normal(0, 1, len(x))   # CV ≈ 1 %
    return x, y


@pytest.fixture
def volatile_series():
    rng = np.random.default_rng(42)
    x = np.arange(2000, 2020, dtype=float)
    y = rng.uniform(10, 100, len(x))        # high CV
    return x, y


@pytest.fixture
def trending_series():
    x = np.arange(2010, 2022, dtype=float)
    y = 100.0 + 10.0 * (x - 2010)
    return x, y


# ---------------------------------------------------------------------------
# get_volatility
# ---------------------------------------------------------------------------

class TestGetVolatility:
    def test_stable_series_low_cv(self, stable_series):
        x, y = stable_series
        cv = InsightExtractor(x, y).get_volatility()
        assert cv < 5.0

    def test_volatile_series_high_cv(self, volatile_series):
        x, y = volatile_series
        cv = InsightExtractor(x, y).get_volatility()
        assert cv > 15.0

    def test_cv_is_float(self, stable_series):
        x, y = stable_series
        cv = InsightExtractor(x, y).get_volatility()
        assert isinstance(cv, float)

    def test_zero_mean_returns_nan(self):
        x = np.arange(5, dtype=float)
        y = np.zeros(5)
        cv = InsightExtractor(x, y).get_volatility()
        assert math.isnan(cv)

    def test_constant_series_zero_cv(self):
        x = np.arange(2010, 2022, dtype=float)
        y = np.full(12, 50.0)
        cv = InsightExtractor(x, y).get_volatility()
        assert cv == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_structural_segments
# ---------------------------------------------------------------------------

class TestGetStructuralSegments:
    def test_returns_list(self, trending_series):
        x, y = trending_series
        segments = InsightExtractor(x, y).get_structural_segments()
        assert isinstance(segments, list)

    def test_trending_series_segments(self, trending_series):
        x, y = trending_series
        segments = InsightExtractor(x, y).get_structural_segments()
        # May be empty (no valid fit) or list of dicts
        assert all(isinstance(s, dict) for s in segments)


# ---------------------------------------------------------------------------
# extract_full_suite
# ---------------------------------------------------------------------------

class TestExtractFullSuite:
    def test_keys_present(self, stable_series):
        x, y = stable_series
        result = InsightExtractor(x, y).extract_full_suite()
        assert "cv_value" in result
        assert "segments" in result

    def test_cv_value_is_numeric(self, stable_series):
        x, y = stable_series
        result = InsightExtractor(x, y).extract_full_suite()
        assert isinstance(result["cv_value"], float)

    def test_segments_is_list(self, stable_series):
        x, y = stable_series
        result = InsightExtractor(x, y).extract_full_suite()
        assert isinstance(result["segments"], list)

    def test_custom_detector_used(self, trending_series):
        """Passing a max_segments=1 detector limits segment count."""
        from trend_narrative.detector import TrendDetector
        x, y = trending_series
        detector = TrendDetector(max_segments=1)
        result = InsightExtractor(x, y, detector=detector).extract_full_suite()
        assert len(result["segments"]) <= 1

    def test_array_like_inputs_accepted(self):
        x = list(range(2010, 2022))
        y = [float(i ** 2) for i in range(12)]
        result = InsightExtractor(x, y).extract_full_suite()
        assert "cv_value" in result

    def test_unsorted_input_sorted_internally(self):
        """Unsorted input should produce same result as sorted input."""
        x_sorted = np.arange(2010, 2022, dtype=float)
        y_sorted = 100.0 + 10.0 * (x_sorted - 2010)

        x_unsorted = x_sorted[::-1]
        y_unsorted = y_sorted[::-1]

        sorted_result = InsightExtractor(x_sorted, y_sorted).extract_full_suite()
        unsorted_result = InsightExtractor(x_unsorted, y_unsorted).extract_full_suite()

        assert sorted_result["cv_value"] == pytest.approx(unsorted_result["cv_value"])
        assert len(sorted_result["segments"]) == len(unsorted_result["segments"])


# ---------------------------------------------------------------------------
# Small dataset handling (0-3 data points)
# ---------------------------------------------------------------------------

class TestSmallDatasetHandling:
    def test_two_points_returns_single_segment(self):
        """Two data points should produce a simple single segment."""
        x = np.array([2020, 2022], dtype=float)
        y = np.array([100, 150], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert len(segments) == 1
        assert segments[0]["start_year"] == 2020
        assert segments[0]["end_year"] == 2022
        assert segments[0]["start_value"] == 100
        assert segments[0]["end_value"] == 150
        assert segments[0]["slope"] == pytest.approx(25.0)
        assert segments[0]["p_value"] is None

    def test_three_points_returns_segment(self):
        """Three data points should produce at least one segment."""
        x = np.array([2018, 2020, 2022], dtype=float)
        y = np.array([80, 100, 120], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert len(segments) >= 1
        assert segments[0]["start_year"] == 2018

    def test_one_point_returns_empty(self):
        """One data point cannot form a segment."""
        x = np.array([2020], dtype=float)
        y = np.array([100], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert segments == []

    def test_empty_returns_empty(self):
        """Empty input returns empty segments."""
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert segments == []

    def test_two_points_decreasing(self):
        """Two decreasing points should have negative slope."""
        x = np.array([2020, 2022], dtype=float)
        y = np.array([200, 100], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert len(segments) == 1
        assert segments[0]["slope"] == pytest.approx(-50.0)

    def test_two_points_same_value(self):
        """Two points with same value should have zero slope."""
        x = np.array([2020, 2022], dtype=float)
        y = np.array([100, 100], dtype=float)
        segments = InsightExtractor(x, y).get_structural_segments()
        assert len(segments) == 1
        assert segments[0]["slope"] == pytest.approx(0.0)
