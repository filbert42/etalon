import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, NamedTuple, Dict, NewType, Union
from scipy.stats.stats import _attempt_exact_2kssamp
from scipy.stats import distributions
from math import gcd
from sklearn.metrics import normalized_mutual_info_score

from mlops_monitoring.signature import Signature


class MetricResult(NamedTuple):
    metric_name: str
    value: float
    passed: bool


# Metrics


def calculate_histogram_intersection(
    signature: Signature, standard: Signature, colname: str, threshold: float = 0.75
) -> MetricResult:
    """Calculate simple metric for histogram intersection between two signatures for given numeric column.

    Metric considered passed if histogram intersection is bigger than the given threshold.

    Args:
        signature: A Signature object, by convention contains profile of the new data
        standard: Another Signature object, by convention contains profile with project standard
        colname: Column name to calculate metric for.
        threshold: Value that defines passed or failed test.

    Returns:
        A MetricResult object that contains a value for metric and a flag if test passed or no.
    """
    # workaround for bug that causes segfault when pmf called on signature col with all NaNs
    if signature.profile.columns[colname].number_tracker.histogram.get_n() == 0:
        return MetricResult("Histogram Intersection", 0.0, False)

    bins, pmf_standard, pmf_signature = get_pmfs(signature, standard, colname)
    hist_intersection = sum(
        min(pmf_standard[i], pmf_signature[i]) for i in range(len(bins))
    )
    passed = hist_intersection >= threshold
    return MetricResult("Histogram Intersection", round(hist_intersection, 2), passed)


def calculate_mutual_info(
    signature: Signature, standard: Signature, colname: str, threshold: float = 0.7
) -> MetricResult:
    """Calculate normalized mutual info between two signatures for given numeric column.

    Metric considered passed if bigger than the given thershold.

    Args:
        See calculate_histogram_intersection()

    Returns:
        A MetricResult object that contains a value for metric and a flag if test passed or no.
    """
    _, pmf_standard, pmf_signature = get_pmfs(signature, standard, colname)
    mutual_info_score = normalized_mutual_info_score(pmf_standard, pmf_signature)
    passed = mutual_info_score >= threshold
    return MetricResult("Normalized Mutual Info", round(mutual_info_score, 2), passed)


def calculate_null_rate_discrepancy(
    signature: Signature, standard: Signature, colname: str, threshold: float = 0.05
) -> MetricResult:
    """Calculate difference between rate of NaN values in signature and standard for given numeric column.

    It's OK if signature has less NaNs than standard, but not otherwise. Considered passed if difference less than the given threshold.

    Args:
        See calculate_histogram_intersection()

    Returns:
        A MetricResult object that contains a value for metric and a flag if test passed or no.
    """
    comparison = create_column_summary_comparison(signature, standard, colname)
    null_rates = get_comparison_metric(comparison, "null_rate")
    null_rate_discrepancy = null_rates["signature"] - null_rates["standard"]
    # it's okay if in signature less nulls than in standard, but not other way
    passed = null_rate_discrepancy <= threshold
    return MetricResult(
        "Null Rate Discrepancy", round(null_rate_discrepancy, 2), passed
    )


def calculate_category_histogram_intersection(
    signature: Signature, standard: Signature, colname: str, threshold: float = 0.75
) -> MetricResult:
    """Calculate simple metric for histogram intersection between two signatures for given category column.

    Metric considered passed if histogram intersection bigger than given the thershold.

    Args:
        See calculate_histogram_intersection()

    Returns:
        A MetricResult object that contains a value for metric and a flag if test passed or no.
    """
    signature_pmf = get_category_pmf(signature, colname)
    standard_pmf = get_category_pmf(standard, colname)
    hist_intersection = sum(
        min(signature_pmf.get(k, 0), standard_pmf.get(k, 0))
        for k in standard_pmf.keys()
    )
    passed = hist_intersection >= threshold
    return MetricResult(
        "Category Histogram Intersection", round(hist_intersection, 2), passed
    )


def calculate_ks_stat(
    signature: Signature, standard: Signature, colname: str, threshold: float = 0.1
) -> MetricResult:
    """Calculate Kolmogorov-Smirnov test for distributions of two signatures for given numeric column.

    Metric considered passed if p-value is bigger than the given threshold.

    Args:
        See calculate_histogram_intersection()

    Returns:
        A MetricResult object that contains a value for metric and a flag if test passed or no.
    """
    n1 = signature.profile.columns[colname].number_tracker.histogram.get_n()
    n2 = standard.profile.columns[colname].number_tracker.histogram.get_n()
    _, cdf_standard, cdf_signature = get_cdfs(signature, standard, colname)
    cddiffs = cdf_signature - cdf_standard

    minS = np.clip(-np.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = np.max(cddiffs)
    d = max(minS, maxS)
    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -np.inf
    mode = "exact" if max(n1, n2) <= 10000 else "asymp"

    if mode == "exact":
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int_).max / n2g:
            mode = "asymp"

    if mode == "exact":
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, "two-sided")
        if not success:
            mode = "asymp"

    if mode == "asymp":

        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        prob = distributions.kstwo.sf(d, np.round(en))

    prob = np.clip(prob, 0, 1)

    passed = prob >= threshold
    return MetricResult("Kolmogorov-Smirnov", round(prob, 2), passed)


# Helpers


def get_pmfs(signature: Signature, standard: Signature, colname: str):
    """Helper to unpack PMFs from the signature objects and given column.

    Args:
        signature: A Signature object, by convention contains profile of the new data
        standard: Another Signature object, by convention contains profile with project standard
        colname: Column name to calculate metric for.

    Returns:
        Bins that are used to generate PMFs, PMF for standard, PMF for Signature.
    """
    bins = get_histogram_bins(signature, standard, colname, 100)
    pmf_standard = np.array(
        standard.profile.columns[colname].number_tracker.histogram.get_pmf(bins[:-1])
    )
    pmf_signature = np.array(
        signature.profile.columns[colname].number_tracker.histogram.get_pmf(bins[:-1])
    )
    return bins, pmf_standard, pmf_signature


def get_cdfs(
    signature: Signature, standard: Signature, colname: str
) -> Tuple[range, np.array, np.array]:
    """Helper to unpack CDFs from the signature objects and given column.

    Args:
        signature: A Signature object, by convention contains profile of the new data
        standard: Another Signature object, by convention contains profile with project standard
        colname: Column name to calculate metric for.

    Returns:
        Bins that are used to generate CDF, CDF for standard, CDF for Signature.
    """
    bins = get_histogram_bins(signature, standard, colname, 100)
    cdf_signature = np.array(
        signature.profile.columns[colname].number_tracker.histogram.get_cdf(bins[:-1])
    )
    cdf_standard = np.array(
        standard.profile.columns[colname].number_tracker.histogram.get_cdf(bins[:-1])
    )
    return bins, cdf_standard, cdf_signature


def get_category_pmf(signature: Signature, colname: str) -> Dict[str, float]:
    counts = signature.profile.flat_summary()["frequent_strings"][colname]
    total_sum = sum(counts.values())
    return {k: v / total_sum for k, v in counts.items()}


def get_histogram_bins(
    signature: Signature, standard: Signature, colname: str, n_bins: int
) -> np.ndarray:

    standard_summary = standard.profile.flat_summary()["summary"]
    signature_summary = signature.profile.flat_summary()["summary"]
    min_range = min(
        standard_summary.query("column == @colname")["min"].values[0],
        signature_summary.query("column == @colname")["min"].values[0],
    )
    max_range = max(
        standard_summary.query("column == @colname")["max"].values[0],
        signature_summary.query("column == @colname")["max"].values[0],
    )

    bins = np.linspace(min_range, max_range, n_bins)
    return bins


def create_column_summary_comparison(
    signature: Signature, standard: Signature, colname: str
) -> pd.DataFrame:
    """Create a DataFrame with column summaries for standard and signature."""

    standard_summary = extract_column_summary(standard, colname).rename("standard")
    signature_summary = extract_column_summary(signature, colname).rename("signature")
    return pd.concat([standard_summary, signature_summary], axis=1)


def extract_column_summary(signature: Signature, colname: str) -> pd.Series:
    """Extract column summary from the signature and reshape it to the long form."""
    signature_summary = signature.profile.flat_summary()["summary"].assign(
        null_rate=lambda x: x.type_null_count / x["count"]
    )
    column_summary = (
        signature_summary.query("column == @colname").unstack().unstack().iloc[:, 0]
    )
    return column_summary


def get_comparison_metric(
    comparison: pd.DataFrame, metric_name: str
) -> Dict[str, float]:
    """Extract specific metric as a dictionary from the DataFrame generated by create_column_summary_comparison() function"""
    return comparison.loc[metric_name].to_dict()
