"""Unit tests for trend_narrative.relationship_narrative."""

import numpy as np
import pytest

from trend_narrative.relationship_analysis import analyze_relationship
from trend_narrative.relationship_narrative import get_relationship_narrative


def _seg(start_year, end_year, slope, start_value=100.0, end_value=None):
    if end_value is None:
        end_value = start_value + slope * (end_year - start_year)
    return {
        "start_year": float(start_year),
        "end_year": float(end_year),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "slope": float(slope),
        "p_value": 0.01,
    }


# ---------------------------------------------------------------------------
# get_relationship_narrative - insufficient data
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeInsufficientData:
    years = np.array([2010, 2020])
    values = np.array([100, 150])
    enough_years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016])
    enough_values = np.array([50, 58, 62, 72, 75, 88, 90], dtype=float)
    empty_years = np.array([], dtype=float)
    empty_values = np.array([], dtype=float)
    expected_narrative = (
        "The relationship between spending and outcome cannot be "
        "determined due to limited data availability."
    )

    def test_too_few_comparison_points(self):
        result = get_relationship_narrative(
            reference_years=self.years,
            reference_values=self.values,
            comparison_years=self.years,
            comparison_values=self.values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert result["narrative"] == self.expected_narrative

    def test_empty_comparison_array(self):
        result = get_relationship_narrative(
            reference_years=self.enough_years,
            reference_values=self.enough_values,
            comparison_years=self.empty_years,
            comparison_values=self.empty_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert result["narrative"] == self.expected_narrative

    def test_empty_reference_array(self):
        result = get_relationship_narrative(
            reference_years=self.empty_years,
            reference_values=self.empty_values,
            comparison_years=self.enough_years,
            comparison_values=self.enough_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert result["narrative"] == self.expected_narrative


# ---------------------------------------------------------------------------
# get_relationship_narrative - comovement path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeComovement:
    years_3pt = np.array([2010, 2015, 2020])
    ref_increasing = np.array([100, 125, 150], dtype=float)
    ref_up_down = np.array([100, 125, 110], dtype=float)
    comp_years_3pt = np.array([2012, 2015, 2018])
    comp_increasing = np.array([50, 65, 80], dtype=float)
    comp_decreasing = np.array([80, 65, 50], dtype=float)
    comp_years_4pt = np.array([2011, 2014, 2016, 2019])
    comp_4pt_increasing = np.array([50, 60, 65, 70], dtype=float)
    two_segments = [_seg(2010, 2015, slope=5), _seg(2015, 2020, slope=-3)]

    def test_single_segment_both_increasing(self):
        result = get_relationship_narrative(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.comp_years_3pt,
            comparison_values=self.comp_increasing,
            reference_name="health spending",
            comparison_name="UHC index",
        )
        assert result["method"] == "comovement"
        assert result["narrative"] == (
            "From 2010 to 2020, health spending increased (100.00 to 150.00) "
            "while UHC index increased (50.00 to 80.00), both moving in the same direction. "
            "With limited UHC index data, a statistical relationship cannot be established."
        )

    def test_single_segment_opposite_directions(self):
        result = get_relationship_narrative(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.comp_years_3pt,
            comparison_values=self.comp_decreasing,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert result["narrative"] == (
            "From 2010 to 2020, spending increased (100.00 to 150.00) "
            "while outcome decreased (80.00 to 50.00), moving in opposite directions. "
            "With limited outcome data, a statistical relationship cannot be established."
        )

    def test_multiple_segments(self):
        result = get_relationship_narrative(
            reference_years=self.years_3pt,
            reference_values=self.ref_up_down,
            comparison_years=self.comp_years_4pt,
            comparison_values=self.comp_4pt_increasing,
            reference_name="spending",
            comparison_name="outcome",
            reference_segments=self.two_segments,
        )
        assert result["method"] == "comovement"
        assert len(result["segment_details"]) == 2
        assert result["narrative"] == (
            "From 2010 to 2015, spending increased (100.00 to 125.00) "
            "while outcome increased (50.00 to 62.50), both moving in the same direction. "
            "From 2015 to 2020, spending decreased (100.00 to 85.00) "
            "while outcome increased (62.50 to 70.00), moving in opposite directions. "
            "With limited outcome data, a statistical relationship cannot be established."
        )

    def test_segment_with_no_comparison_data(self):
        """Segments without comparison data are omitted, with a caveat added."""
        comp_years = np.array([2011, 2013, 2014])
        comp_values = np.array([50, 55, 60], dtype=float)
        result = get_relationship_narrative(
            reference_years=self.years_3pt,
            reference_values=self.ref_up_down,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_segments=self.two_segments,
        )
        assert result["method"] == "comovement"
        assert result["narrative"] == (
            "From 2010 to 2015, spending increased (100.00 to 125.00) "
            "while outcome increased (50.00 to 60.00), both moving in the same direction. "
            "With limited outcome data, a statistical relationship cannot be established."
        )

    def test_no_comparison_data_in_segments(self):
        """When comparison has zero observations in all segments, simplify narrative."""
        ref_years = np.array([2019, 2020, 2021, 2022])
        ref_values = np.array([100, 110, 105, 115], dtype=float)
        comp_years = np.array([2010, 2012, 2015, 2017])
        comp_values = np.array([50, 55, 60, 65], dtype=float)
        segments = [_seg(2019, 2021, slope=2.5), _seg(2021, 2022, slope=10)]
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_segments=segments,
        )
        assert result["method"] == "comovement"
        assert result["narrative"] == (
            "The relationship between spending and outcome "
            "cannot be determined because outcome data is not available."
        )

    def test_remained_stable_when_formatted_values_same(self):
        """When formatted start/end values are equal, direction should be 'remained stable'."""
        ref_years = np.array([2018, 2020, 2022, 2024])
        ref_values = np.array([100.001, 100.002, 100.003, 100.004], dtype=float)
        comp_years = np.array([2019, 2021, 2023])
        comp_values = np.array([50, 55, 60], dtype=float)
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert "remained stable (100.00 to 100.00)" in result["narrative"]
        assert "increased (100.00 to 100.00)" not in result["narrative"]

    def test_handles_unsorted_input(self):
        """Input data should be sorted internally, producing same result as sorted input."""
        sorted_result = get_relationship_narrative(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.comp_years_3pt,
            comparison_values=self.comp_increasing,
            reference_name="spending",
            comparison_name="outcome",
        )
        unsorted_result = get_relationship_narrative(
            reference_years=self.years_3pt[::-1],
            reference_values=self.ref_increasing[::-1],
            comparison_years=self.comp_years_3pt[::-1],
            comparison_values=self.comp_increasing[::-1],
            reference_name="spending",
            comparison_name="outcome",
        )
        assert sorted_result["narrative"] == unsorted_result["narrative"]
        assert sorted_result["method"] == unsorted_result["method"]

    def test_comparison_all_same_value(self):
        """When all comparison values are identical, should say 'remained stable' not 'only one observation'."""
        ref_years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, 125, 150], dtype=float)
        comp_years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017])
        comp_values = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert "remained stable" in result["narrative"]
        assert "only one" not in result["narrative"]


# ---------------------------------------------------------------------------
# get_relationship_narrative - lagged correlation path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeLaggedCorrelation:
    periods_10 = np.arange(1, 11)
    periods_15 = np.arange(1, 16)
    ref_lag0_pos = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
    comp_lag0_pos = np.array([50, 55, 58, 65, 68, 75, 78, 85, 90, 98], dtype=float)
    ref_lag0_neg = np.array([100, 110, 115, 130, 140, 145, 160, 175, 180, 200], dtype=float)
    comp_lag0_neg = np.array([100, 95, 93, 85, 80, 78, 70, 62, 60, 50], dtype=float)
    ref_lag0_insig = np.array([100, 105, 102, 108, 104, 110, 106, 112, 108, 114], dtype=float)
    comp_lag0_insig = np.array([50, 48, 52, 49, 51, 47, 53, 50, 48, 52], dtype=float)
    ref_lag2 = np.array([100, 100, 110, 110, 120, 120, 130, 130, 140, 140, 150, 150, 160, 160, 170], dtype=float)
    comp_lag2 = np.array([50, 50, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80], dtype=float)
    ref_lag1 = np.array([100, 110, 110, 120, 120, 130, 130, 140, 140, 150, 150, 160, 160, 170, 170], dtype=float)
    comp_lag1 = np.array([50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80, 85], dtype=float)

    def test_positive_correlation(self):
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(0.975, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.0, abs=0.001)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "When spending increases, outcome tends to increase in the same year. "
            "This is a very strong relationship (r=0.98) and is statistically reliable (p=0.000), "
            "based on 9 year-over-year comparisons."
        )

    def test_negative_correlation(self):
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_neg,
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_neg,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(-0.742, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.022, rel=0.1)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "When spending increases, outcome tends to decrease in the same year. "
            "This is a very strong relationship (r=-0.74) and is statistically reliable (p=0.022), "
            "based on 9 year-over-year comparisons."
        )

    def test_insignificant_correlation(self):
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_insig,
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_insig,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(-0.566, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.112, rel=0.1)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "No reliable relationship was detected between changes in spending and outcome. "
            "While the data suggests a strong negative pattern (r=-0.57), this could be due to chance "
            "given the limited sample size (n=9 change pairs, p=0.11)."
        )

    def test_falls_back_to_comovement_below_threshold(self):
        years = np.array([2010, 2012, 2015, 2018])
        ref_values = np.array([100, 110, 125, 140], dtype=float)
        comp_values = np.array([50, 55, 62, 70], dtype=float)
        result = get_relationship_narrative(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
        )
        assert result["method"] == "comovement"

    def test_falls_back_to_comovement_when_no_valid_correlations(self):
        """When enough points but no valid correlations computed, fall back to comovement."""
        ref_years = np.array([2010, 2011, 2012, 2013, 2014, 2015])
        ref_values = np.array([100, 110, 120, 130, 140, 150], dtype=float)
        comp_years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
        comp_values = np.array([50, 60, 70, 80, 90, 100], dtype=float)
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
        )
        assert result["method"] == "comovement"
        assert result["best_lag"] is None
        assert result["all_lags"] is None

    def test_reference_sparser_narrative_reflects_comparison_leading(self):
        """When reference is sparser, narrative says comparison leads reference."""
        result = get_relationship_narrative(
            reference_years=self.periods_10[:5],
            reference_values=self.ref_lag0_pos[:5],
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
        )
        assert result["method"] == "lagged_correlation"
        assert "When outcome increases" in result["narrative"]
        assert "spending tends to" in result["narrative"]

    def test_comparison_sparser_narrative_reflects_reference_leading(self):
        """When comparison is sparser, narrative says reference leads comparison."""
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10[:5],
            comparison_values=self.comp_lag0_pos[:5],
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
        )
        assert result["method"] == "lagged_correlation"
        assert "When spending increases" in result["narrative"]
        assert "outcome tends to" in result["narrative"]

    def test_reference_leads_override_true(self):
        """User can force reference_leads=True even when reference is sparser."""
        result = get_relationship_narrative(
            reference_years=self.periods_10[:5],
            reference_values=self.ref_lag0_pos[:5],
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
            reference_leads=True,
        )
        assert result["method"] == "lagged_correlation"
        assert "When spending increases" in result["narrative"]
        assert "outcome tends to" in result["narrative"]

    def test_reference_leads_override_false(self):
        """User can force reference_leads=False even when comparison is sparser."""
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10[:5],
            comparison_values=self.comp_lag0_pos[:5],
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
            reference_leads=False,
        )
        assert result["method"] == "lagged_correlation"
        assert "When outcome increases" in result["narrative"]
        assert "spending tends to" in result["narrative"]

    def test_lagged_effect_detection(self):
        """Test that lagged effects can be detected."""
        result = get_relationship_narrative(
            reference_years=self.periods_15,
            reference_values=self.ref_lag2,
            comparison_years=self.periods_15,
            comparison_values=self.comp_lag2,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
            max_lag_cap=5,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 2
        assert result["best_lag"]["correlation"] == pytest.approx(1.0, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.0, abs=0.001)
        assert result["best_lag"]["n_pairs"] == 12
        assert result["all_lags"] is not None
        assert len(result["all_lags"]) == 6
        assert result["max_lag_tested"] == 5
        assert result["narrative"] == (
            "When spending increases, outcome tends to increase about 2 years later. "
            "This is a very strong relationship (r=1.00) and is statistically reliable (p=0.000), "
            "based on 12 year-over-year comparisons."
        )

    def test_max_lag_zero_no_relationship(self):
        """When max_lag is 0 and no relationship found, narrative should not say '0-0 years'."""
        periods_5 = np.array([2010, 2011, 2012, 2013, 2014])
        ref_values = np.array([100, 110, 100, 110, 100], dtype=float)
        comp_values = np.array([50, 60, 70, 60, 50], dtype=float)
        result = get_relationship_narrative(
            reference_years=periods_5,
            reference_values=ref_values,
            comparison_years=periods_5,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=5,
        )
        assert result["method"] == "lagged_correlation"
        assert result["max_lag_tested"] == 0
        assert result["narrative"] == (
            "No reliable relationship was detected between changes in spending and outcome. "
            "Changes in one do not appear to be associated with changes in the other, "
            "based on 4 year-over-year comparisons."
        )

    def test_time_unit_month(self):
        result = get_relationship_narrative(
            reference_years=self.periods_15,
            reference_values=self.ref_lag2,
            comparison_years=self.periods_15,
            comparison_values=self.comp_lag2,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
            time_unit="month",
        )
        assert result["method"] == "lagged_correlation"
        assert "2 months later" in result["narrative"]
        assert "month-over-month comparisons" in result["narrative"]

    def test_time_unit_quarter(self):
        result = get_relationship_narrative(
            reference_years=self.periods_15,
            reference_values=self.ref_lag2,
            comparison_years=self.periods_15,
            comparison_values=self.comp_lag2,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
            time_unit="quarter",
        )
        assert result["method"] == "lagged_correlation"
        assert "2 quarters later" in result["narrative"]
        assert "quarter-over-quarter comparisons" in result["narrative"]

    def test_time_unit_lag_one_singular(self):
        result = get_relationship_narrative(
            reference_years=self.periods_15,
            reference_values=self.ref_lag1,
            comparison_years=self.periods_15,
            comparison_values=self.comp_lag1,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
            time_unit="month",
        )
        assert result["method"] == "lagged_correlation"
        assert "1 month later" in result["narrative"]

    def test_time_unit_same_period(self):
        result = get_relationship_narrative(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
            time_unit="quarter",
        )
        assert result["method"] == "lagged_correlation"
        assert "in the same quarter" in result["narrative"]


# ---------------------------------------------------------------------------
# get_relationship_narrative - NaN handling
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeNanHandling:
    def test_removes_nan_from_comparison(self):
        ref_years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, 125, 150])
        comp_years = np.array([2012, 2015, 2018, 2019])
        comp_values = np.array([50, np.nan, 80, 90])
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["n_points"] == 3


# ---------------------------------------------------------------------------
# get_relationship_narrative - return structure
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeReturnStructure:
    def test_comovement_return_keys(self):
        ref_years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, 125, 150])
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([50, 65, 80])
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "narrative" in result
        assert "method" in result
        assert "n_points" in result
        assert "segment_details" in result
        assert "best_lag" in result
        assert "all_lags" in result
        assert "max_lag_tested" in result
        assert result["best_lag"] is None
        assert result["all_lags"] is None

    def test_lagged_correlation_return_keys(self):
        years = np.arange(2010, 2020)
        ref_values = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
        comp_values = np.array([50, 55, 58, 65, 68, 75, 78, 85, 90, 98], dtype=float)

        result = get_relationship_narrative(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            correlation_threshold=8,
        )
        assert "narrative" in result
        assert "method" in result
        assert "n_points" in result
        assert "best_lag" in result
        assert "all_lags" in result
        assert "max_lag_tested" in result
        assert result["best_lag"] is not None
        assert "lag" in result["best_lag"]
        assert "correlation" in result["best_lag"]
        assert "p_value" in result["best_lag"]
        assert "n_pairs" in result["best_lag"]

    def test_insufficient_data_return_keys(self):
        ref_years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, 125, 150])
        comp_years = np.array([2012, 2015])
        comp_values = np.array([50, 60])
        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert result["segment_details"] is None
        assert result["best_lag"] is None
        assert result["all_lags"] is None
        assert result["max_lag_tested"] is None


# ---------------------------------------------------------------------------
# get_relationship_narrative - number formatting
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeNumberFormatting:
    ref_years = np.array([2010, 2015, 2020])
    ref_values = np.array([1000.55, 1250.75, 1500.25])
    comp_years = np.array([2012, 2015, 2018])
    comp_values = np.array([0.5055, 0.6575, 0.8025])

    def test_default_format(self):
        result = get_relationship_narrative(
            reference_years=self.ref_years,
            reference_values=self.ref_values,
            comparison_years=self.comp_years,
            comparison_values=self.comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "(1000.55 to 1500.25)" in result["narrative"]
        assert "(0.51 to 0.80)" in result["narrative"]

    def test_custom_reference_format(self):
        result = get_relationship_narrative(
            reference_years=self.ref_years,
            reference_values=self.ref_values,
            comparison_years=self.comp_years,
            comparison_values=self.comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_format=",.0f",
        )
        assert "(1,001 to 1,500)" in result["narrative"]
        assert "(0.51 to 0.80)" in result["narrative"]

    def test_custom_comparison_format(self):
        result = get_relationship_narrative(
            reference_years=self.ref_years,
            reference_values=self.ref_values,
            comparison_years=self.comp_years,
            comparison_values=self.comp_values,
            reference_name="spending",
            comparison_name="outcome",
            comparison_format=".1%",
        )
        assert "(1000.55 to 1500.25)" in result["narrative"]
        assert "(50.5% to 80.2%)" in result["narrative"]

    def test_separate_formats(self):
        result = get_relationship_narrative(
            reference_years=self.ref_years,
            reference_values=self.ref_values,
            comparison_years=self.comp_years,
            comparison_values=self.comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_format=",.0f",
            comparison_format=".1%",
        )
        assert "(1,001 to 1,500)" in result["narrative"]
        assert "(50.5% to 80.2%)" in result["narrative"]

    def test_callable_formatter(self):
        """Callable formatters should work for custom formatting like currency."""
        def currency_fmt(x):
            if x >= 1000:
                return f"USD {x/1000:.1f}K"
            return f"USD {x:.0f}"

        result = get_relationship_narrative(
            reference_years=self.ref_years,
            reference_values=self.ref_values,
            comparison_years=self.comp_years,
            comparison_values=self.comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_format=currency_fmt,
        )
        assert "(USD 1.0K to USD 1.5K)" in result["narrative"]


# ---------------------------------------------------------------------------
# get_relationship_narrative - precomputed insights path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativePrecomputedPath:
    def test_comovement_from_insights(self):
        """Path 2: generate narrative from precomputed comovement insights."""
        insights = {
            "method": "comovement",
            "n_points": 3,
            "segment_details": [
                {
                    "start_year": 2010,
                    "end_year": 2020,
                    "reference_direction": "increased",
                    "reference_start": 100.0,
                    "reference_end": 150.0,
                    "comparison_n_points": 3,
                    "comparison_direction": "increased",
                    "comparison_start": 50.0,
                    "comparison_end": 80.0,
                    "interpolated": False,
                }
            ],
            "best_lag": None,
            "all_lags": None,
            "max_lag_tested": None,
            "reference_leads": True,
        }
        result = get_relationship_narrative(
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert "spending" in result["narrative"]
        assert "outcome" in result["narrative"]
        assert "both moving in the same direction" in result["narrative"]

    def test_lagged_correlation_from_insights(self):
        """Path 2: generate narrative from precomputed correlation insights."""
        insights = {
            "method": "lagged_correlation",
            "n_points": 10,
            "segment_details": None,
            "best_lag": {
                "lag": 0,
                "correlation": 0.98,
                "p_value": 0.001,
                "n_pairs": 9,
            },
            "all_lags": [
                {"lag": 0, "correlation": 0.98, "p_value": 0.001, "n_pairs": 9},
            ],
            "max_lag_tested": 0,
            "reference_leads": True,
        }
        result = get_relationship_narrative(
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "lagged_correlation"
        assert "spending" in result["narrative"]
        assert "outcome" in result["narrative"]
        assert "very strong" in result["narrative"]

    def test_insufficient_data_from_insights(self):
        """Path 2: generate narrative from precomputed insufficient data insights."""
        insights = {
            "method": "insufficient_data",
            "n_points": 2,
            "segment_details": None,
            "best_lag": None,
            "all_lags": None,
            "max_lag_tested": None,
            "reference_leads": True,
        }
        result = get_relationship_narrative(
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert "cannot be determined" in result["narrative"]

    def test_reference_leads_override_with_insights(self):
        """User can override reference_leads even when using precomputed insights."""
        insights = {
            "method": "lagged_correlation",
            "n_points": 10,
            "segment_details": None,
            "best_lag": {
                "lag": 0,
                "correlation": 0.98,
                "p_value": 0.001,
                "n_pairs": 9,
            },
            "all_lags": [
                {"lag": 0, "correlation": 0.98, "p_value": 0.001, "n_pairs": 9},
            ],
            "max_lag_tested": 0,
            "reference_leads": True,
        }
        result = get_relationship_narrative(
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
            reference_leads=False,
        )
        assert "When outcome increases" in result["narrative"]
        assert "spending tends to" in result["narrative"]

    def test_raises_without_insights_or_data(self):
        """Should raise ValueError if neither insights nor data provided."""
        with pytest.raises(ValueError, match="Provide either insights="):
            get_relationship_narrative(
                reference_name="spending",
                comparison_name="outcome",
            )

    def test_insights_ignores_data_arrays(self):
        """When insights provided, data arrays are ignored."""
        insights = {
            "method": "insufficient_data",
            "n_points": 2,
            "segment_details": None,
            "best_lag": None,
            "all_lags": None,
            "max_lag_tested": None,
            "reference_leads": True,
        }
        result = get_relationship_narrative(
            reference_years=np.arange(2010, 2025),
            reference_values=np.arange(15, dtype=float) * 10,
            comparison_years=np.arange(2010, 2025),
            comparison_values=np.arange(15, dtype=float) * 5,
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"

    def test_roundtrip_analyze_then_narrative(self):
        """analyze_relationship output can be passed directly to get_relationship_narrative."""
        years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, 125, 150], dtype=float)
        comp_values = np.array([50, 65, 80], dtype=float)

        insights = analyze_relationship(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
        )
        result = get_relationship_narrative(
            insights=insights,
            reference_name="spending",
            comparison_name="outcome",
        )

        direct_result = get_relationship_narrative(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )

        assert result["narrative"] == direct_result["narrative"]
        assert result["method"] == direct_result["method"]
