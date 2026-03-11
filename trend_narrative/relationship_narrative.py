"""
trend_narrative.relationship_narrative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate human-readable narratives from relationship analysis results.

This module handles the text generation layer, converting structured
analysis outputs into plain-English descriptions.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from .relationship_analysis import (
    analyze_relationship,
    get_correlation_strength,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_MAX_LAG_CAP,
    P_THRESHOLD,
)

Formatter = Union[str, Callable[[float], str]]


def _format_value(value: float, fmt: Formatter) -> str:
    """Format a numeric value using the given format spec or callable."""
    if callable(fmt):
        return fmt(value)
    return f"{value:{fmt}}"


def _pluralize(word: str, count: int) -> str:
    """Return plural form if count != 1."""
    return word if count == 1 else f"{word}s"


def _build_comovement_narrative(
    segment_details: list[dict],
    reference_name: str,
    comparison_name: str,
    reference_format: Formatter = ".2f",
    comparison_format: Formatter = ".2f",
) -> str:
    """Build narrative from segment-level co-movement analysis."""
    if not segment_details:
        return f"Unable to analyze relationship between {reference_name} and {comparison_name}."

    total_comparison_points = sum(seg["comparison_n_points"] for seg in segment_details)
    if total_comparison_points == 0:
        return (
            f"The relationship between {reference_name} and {comparison_name} "
            f"cannot be determined because {comparison_name} data is not available."
        )

    # Only build narratives for segments with comparison data
    narratives = []

    for seg in segment_details:
        comp_n = seg["comparison_n_points"]
        if comp_n == 0:
            continue

        period = f"from {seg['start_year']} to {seg['end_year']}"
        ref_dir = seg["reference_direction"]
        comp_dir = seg["comparison_direction"]

        ref_start = _format_value(seg["reference_start"], reference_format)
        ref_end = _format_value(seg["reference_end"], reference_format)

        # Override direction if formatted values are the same
        if ref_start == ref_end:
            ref_dir = "remained stable"

        if comp_dir is None:
            if comp_n == 1:
                comp_start = _format_value(seg["comparison_start"], comparison_format)
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} "
                    f"({ref_start} to {ref_end}), "
                    f"with only one {comparison_name} observation ({comp_start})"
                )
            else:
                # Multiple observations but all same value - remained stable
                comp_start = _format_value(seg["comparison_start"], comparison_format)
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} remained stable ({comp_start})"
                )
        else:
            comp_start = _format_value(seg["comparison_start"], comparison_format)
            comp_end = _format_value(seg["comparison_end"], comparison_format)

            # Override direction if formatted values are the same
            if comp_start == comp_end:
                comp_dir = "remained stable"

            # Describe co-movement
            if ref_dir == comp_dir:
                relationship = "both moving in the same direction"
            elif ref_dir == "remained stable" or comp_dir == "remained stable":
                relationship = None
            else:
                relationship = "moving in opposite directions"

            if relationship:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} {comp_dir} "
                    f"({comp_start} to {comp_end}), {relationship}"
                )
            else:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} {comp_dir} "
                    f"({comp_start} to {comp_end})"
                )

        # Capitalize first letter
        seg_narrative = seg_narrative[0].upper() + seg_narrative[1:]
        narratives.append(seg_narrative)

    # Join segments
    if len(narratives) == 1:
        narrative = narratives[0] + "."
    else:
        narrative = ". ".join(narratives) + "."

    # Add caveat about limited data
    narrative += (
        f" With limited {comparison_name} data, "
        "a statistical relationship cannot be established."
    )

    return narrative


def _build_lagged_correlation_narrative(
    best_lag: dict,
    all_lags: list[dict],
    n_sparse: int,
    max_lag_tested: int,
    reference_name: str,
    comparison_name: str,
    time_unit: str = "year",
    reference_leads: bool = True,
) -> str:
    """Build narrative from lagged correlation analysis.

    When reference_leads=True: "When reference increases, comparison follows"
    When reference_leads=False: "When comparison increases, reference follows"
    """
    correlation = best_lag["correlation"]
    p_value = best_lag["p_value"]
    lag = best_lag["lag"]
    n_pairs = best_lag["n_pairs"]

    strength = get_correlation_strength(correlation)
    is_significant = p_value < P_THRESHOLD

    # Determine which series leads based on computation
    if reference_leads:
        leader_name, follower_name = reference_name, comparison_name
    else:
        leader_name, follower_name = comparison_name, reference_name

    # Build lag timing description
    if lag == 0:
        timing = f"in the same {time_unit}"
    else:
        timing = f"about {lag} {_pluralize(time_unit, lag)} later"

    # Not significant: lead with uncertainty
    if strength == "no" or not is_significant:
        narrative = (
            f"No reliable relationship was detected between changes in {reference_name} "
            f"and {comparison_name}. "
        )
        if strength != "no":
            narrative += (
                f"While the data suggests a {strength} {'positive' if correlation > 0 else 'negative'} "
                f"pattern (r={correlation:.2f}), this could be due to chance "
                f"given the limited sample size (n={n_pairs} change pairs, p={p_value:.2f})."
            )
        else:
            lag_info = (
                "" if max_lag_tested == 0
                else f" at any lag tested (0-{max_lag_tested} {_pluralize(time_unit, max_lag_tested)})"
            )
            narrative += (
                f"Changes in one do not appear to be associated with changes in the other{lag_info}, "
                f"based on {n_pairs} {time_unit}-over-{time_unit} comparisons."
            )
    else:
        # Significant: lead with the finding
        direction_word = "increase" if correlation > 0 else "decrease"
        narrative = (
            f"When {leader_name} increases, {follower_name} tends to "
            f"{direction_word} {timing}. "
            f"This is a {strength} relationship (r={correlation:.2f}) "
            f"and is statistically reliable (p={p_value:.3f}), "
            f"based on {n_pairs} {time_unit}-over-{time_unit} comparisons."
        )

    return narrative


def get_relationship_narrative(
    reference_years: "array-like" = None,
    reference_values: "array-like" = None,
    comparison_years: "array-like" = None,
    comparison_values: "array-like" = None,
    reference_name: str = "",
    comparison_name: str = "",
    reference_segments: Optional[list[dict]] = None,
    correlation_threshold: int = DEFAULT_CORRELATION_THRESHOLD,
    max_lag_cap: int = DEFAULT_MAX_LAG_CAP,
    reference_format: Formatter = ".2f",
    comparison_format: Formatter = ".2f",
    time_unit: str = "year",
    reference_leads: Optional[bool] = None,
    insights: Optional[dict] = None,
) -> dict:
    """
    Analyze relationship between two time series and generate narrative.

    Supports two calling paths:

    **Path 1 – from raw data** (analysis computed on the fly):

    .. code-block:: python

        get_relationship_narrative(
            reference_years=years1,
            reference_values=values1,
            comparison_years=years2,
            comparison_values=values2,
            reference_name="spending",
            comparison_name="outcome",
        )

    **Path 2 – precomputed insights** (e.g. insights already stored in a
    Delta table — no re-analysis required):

    .. code-block:: python

        get_relationship_narrative(
            insights=row["relationship_insights"],
            reference_name="spending",
            comparison_name="outcome",
        )

    Parameters
    ----------
    reference_years : array-like, optional
        Year values for reference series (Path 1).
    reference_values : array-like, optional
        Data values for reference series (Path 1).
    comparison_years : array-like, optional
        Year values for the comparison series (Path 1).
    comparison_values : array-like, optional
        Data values for the comparison series (Path 1).
    reference_name : str
        Display name for the reference series.
    comparison_name : str
        Display name for the comparison series.
    reference_segments : list[dict], optional
        Pre-computed segments from InsightExtractor for the reference series.
        Each dict should contain: start_year, end_year, start_value, end_value.
        If not provided, computed from reference_years/reference_values.
    correlation_threshold : int
        Minimum points to use correlation analysis (default 5).
        Below this, comovement analysis is used.
    max_lag_cap : int
        Maximum lag to test in years (default 5). Actual max lag may be
        lower if data is insufficient.
    reference_format : str or callable
        Format spec (e.g., ".2f") or callable (e.g., lambda x: f"${x:,.0f}")
        for reference series values in narratives. Default ".2f".
    comparison_format : str or callable
        Format spec or callable for comparison series values. Default ".2f".
    time_unit : str
        Time unit label for narratives (default "year"). Use "month", "quarter", etc.
    reference_leads : bool, optional
        Controls narrative direction for lagged correlation:
        - True: "When reference increases, comparison follows"
        - False: "When comparison increases, reference follows"
        - None (default): inferred from sparsity (sparser series is the follower)
    insights : dict, optional
        Pre-computed insights from analyze_relationship() (Path 2).
        If provided, raw data arrays are ignored.

    Returns
    -------
    dict
        Keys:
        - narrative: str, human-readable description
        - method: str, one of "insufficient_data", "comovement", "lagged_correlation"
        - n_points: int, number of data points in sparser series
        - segment_details: list[dict], per-segment analysis (comovement only)
        - best_lag: dict with lag, correlation, p_value, n_pairs (correlation only)
        - all_lags: list[dict], results for all tested lags (correlation only)
        - max_lag_tested: int, maximum lag that was tested (correlation only)

    Raises
    ------
    ValueError
        If neither insights nor data arrays are provided.
    """
    if insights is not None:
        analysis = insights
    elif reference_years is not None and comparison_years is not None:
        analysis = analyze_relationship(
            reference_years=reference_years,
            reference_values=reference_values,
            comparison_years=comparison_years,
            comparison_values=comparison_values,
            reference_segments=reference_segments,
            correlation_threshold=correlation_threshold,
            max_lag_cap=max_lag_cap,
        )
    else:
        raise ValueError(
            "Provide either insights= or data arrays "
            "(reference_years, reference_values, comparison_years, comparison_values)"
        )

    if reference_leads is None:
        reference_leads = analysis.get("reference_leads", True)

    method = analysis["method"]
    n_points = analysis["n_points"]

    if method == "insufficient_data":
        narrative = (
            f"The relationship between {reference_name} and {comparison_name} "
            "cannot be determined due to limited data availability."
        )
    elif method == "lagged_correlation":
        narrative = _build_lagged_correlation_narrative(
            analysis["best_lag"],
            analysis["all_lags"],
            n_points,
            analysis["max_lag_tested"],
            reference_name,
            comparison_name,
            time_unit,
            reference_leads=reference_leads,
        )
    else:
        narrative = _build_comovement_narrative(
            analysis["segment_details"],
            reference_name,
            comparison_name,
            reference_format=reference_format,
            comparison_format=comparison_format,
        )

    return {
        "narrative": narrative,
        "method": method,
        "n_points": n_points,
        "segment_details": analysis["segment_details"],
        "best_lag": analysis["best_lag"],
        "all_lags": analysis["all_lags"],
        "max_lag_tested": analysis["max_lag_tested"],
    }
