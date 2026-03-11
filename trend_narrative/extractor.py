"""
trend_narrative.extractor
~~~~~~~~~~~~~~~~~~~~~~~~~
High-level facade that combines volatility measurement with piecewise-linear
trend detection.

Ported from dime-worldbank/mega-boost analytics/insight_extractor.py
"""

from __future__ import annotations

import numpy as np

from .detector import TrendDetector


class InsightExtractor:
    """Extract statistical insights from a univariate time series.

    Parameters
    ----------
    x : array-like
        1-D array of x-values (e.g. integer years).
    y : array-like
        1-D array of observed metric values aligned with *x*.
    detector : TrendDetector, optional
        Custom detector instance.  A default :class:`TrendDetector` is
        used when not provided.

    Examples
    --------
    >>> import numpy as np
    >>> from trend_narrative import InsightExtractor
    >>> x = np.arange(2010, 2022)
    >>> y = np.array([100, 105, 110, 108, 115, 130, 125, 120, 118, 122, 130, 140])
    >>> extractor = InsightExtractor(x, y)
    >>> result = extractor.extract_full_suite()
    >>> "cv_value" in result and "segments" in result
    True
    """

    def __init__(
        self,
        x: "array-like",
        y: "array-like",
        detector: TrendDetector | None = None,
    ) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # Sort by x to ensure correct segment computation
        sort_idx = np.argsort(x)
        self.x = x[sort_idx]
        self.y = y[sort_idx]
        self.trend_detector = detector if detector is not None else TrendDetector()

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def get_volatility(self) -> float:
        """Coefficient of Variation (%) – a measure of relative spread.

        Returns
        -------
        float
            ``(std / mean) * 100``.  Returns *NaN* when the mean is zero.
        """
        mean = self.y.mean()
        if mean == 0:
            return float("nan")
        return float((self.y.std() / mean) * 100)

    def get_structural_segments(self) -> list[dict]:
        """Run the trend detector and return per-segment statistics.

        For 2-3 data points where piecewise fitting isn't possible,
        returns a simple single segment from first to last point.

        Returns
        -------
        list[dict]
            See :meth:`TrendDetector.extract_trend` for the dict schema.
        """
        if len(self.x) < 2:
            return []

        segments = self.trend_detector.extract_trend(self.x, self.y)

        # Fallback for small datasets: create simple start-to-end segment
        if not segments and len(self.x) >= 2:
            start_year, end_year = self.x[0], self.x[-1]
            start_value, end_value = self.y[0], self.y[-1]
            duration = end_year - start_year
            slope = (end_value - start_value) / duration if duration > 0 else 0.0
            segments = [{
                "start_year": float(start_year),
                "end_year": float(end_year),
                "start_value": float(start_value),
                "end_value": float(end_value),
                "slope": float(slope),
                "p_value": None,  # No statistical test for simple segment
            }]

        return segments

    # ------------------------------------------------------------------
    # Convenience bundle
    # ------------------------------------------------------------------

    def extract_full_suite(self) -> dict:
        """Return all insights as a single flat dictionary.

        Keys
        ----
        cv_value : float
            Coefficient of Variation (%).
        segments : list[dict]
            Piecewise-linear segment statistics.
        n_points : int
            Number of data points in the series.
        """
        return {
            "cv_value": self.get_volatility(),
            "segments": self.get_structural_segments(),
            "n_points": len(self.x),
        }
