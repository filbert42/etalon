import pytest
import numpy as np
import pandas as pd

from mlops_monitoring.metrics import *
from mlops_monitoring.signature import new_signature


class TestMetrcis:
    def test_get_histogram_bins(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        bins1 = get_histogram_bins(rand_sig, rand_sig2, "A", 2)
        assert len(bins1) == 2

        bins2 = get_histogram_bins(rand_sig, rand_sig2, "A", 100)
        assert len(bins2) == 100

    # NB: tests for get_cdfs and get_pmfs assume that whylogs creates correct cdf and pmf
    def test_get_cdf(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        bins, sign_cdf, standard_cdf = get_cdfs(rand_sig, rand_sig2, "A")

        assert np.array_equal(bins, get_histogram_bins(rand_sig, rand_sig2, "A", 100))
        assert isinstance(sign_cdf, np.ndarray)
        assert isinstance(standard_cdf, np.ndarray)
        # CDFs of different columns shouldn't be equal
        assert not np.array_equal(sign_cdf, standard_cdf)

        bins, sign_cdf, standard_cdf = get_cdfs(rand_sig, rand_sig, "A")
        # CDFs of the exact same columns should be equal
        assert np.array_equal(sign_cdf, standard_cdf)

    def test_get_pmfs(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        bins, sign_pmf, standard_pmf = get_pmfs(rand_sig, rand_sig2, "A")

        assert np.array_equal(bins, get_histogram_bins(rand_sig, rand_sig2, "A", 100))
        assert isinstance(sign_pmf, np.ndarray)
        assert isinstance(standard_pmf, np.ndarray)
        # CDFs of different columns shouldn't be equal
        assert not np.array_equal(sign_pmf, standard_pmf)

        bins, sign_pmf, standard_pmf = get_pmfs(rand_sig, rand_sig, "A")
        # CDFs of the exact same columns should be equal
        assert np.array_equal(sign_pmf, standard_pmf)

    def test_get_category_pmf(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        pmf = get_category_pmf(mixed_sig, "C")
        assert list(pmf.values()) == [0.2, 0.2, 0.2, 0.2, 0.2]

        mixed_frame_diff = pd.util.testing.makeMixedDataFrame().assign(
            C=["foo1", "foo1", "foo2", "foo3", "foo3"]
        )
        mixed_sig_diff = new_signature(mixed_frame_diff, "project")
        pmf2 = get_category_pmf(mixed_sig_diff, "C")
        assert pmf2 == {"foo1": 0.4, "foo2": 0.2, "foo3": 0.4}

    def test_create_column_summary_comparison(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        comparison = create_column_summary_comparison(rand_sig, rand_sig2, "A")

        assert isinstance(comparison, pd.DataFrame)
        assert comparison.index.to_list() == [
            "column",
            "count",
            "null_count",
            "bool_count",
            "numeric_count",
            "max",
            "mean",
            "min",
            "stddev",
            "nunique_numbers",
            "nunique_numbers_lower",
            "nunique_numbers_upper",
            "inferred_dtype",
            "dtype_fraction",
            "type_unknown_count",
            "type_null_count",
            "type_fractional_count",
            "type_integral_count",
            "type_boolean_count",
            "type_string_count",
            "nunique_str",
            "nunique_str_lower",
            "nunique_str_upper",
            "quantile_0.0000",
            "quantile_0.0100",
            "quantile_0.0500",
            "quantile_0.2500",
            "quantile_0.5000",
            "quantile_0.7500",
            "quantile_0.9500",
            "quantile_0.9900",
            "quantile_1.0000",
            "null_rate",
        ]
        assert comparison.columns.to_list() == ["standard", "signature"]
        assert comparison.loc["column"].to_dict() == {"standard": "A", "signature": "A"}
        # different signatures -> summary cols not equal
        assert not comparison.standard.equals(comparison.signature)

        # same signatures -> summary cols equal
        comparison = create_column_summary_comparison(rand_sig, rand_sig, "A")
        assert comparison.standard.equals(comparison.signature)

    def test_extract_column_summary(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        summary = extract_column_summary(rand_sig, "A")
        assert isinstance(summary, pd.Series)
        assert summary.index.to_list() == [
            "column",
            "count",
            "null_count",
            "bool_count",
            "numeric_count",
            "max",
            "mean",
            "min",
            "stddev",
            "nunique_numbers",
            "nunique_numbers_lower",
            "nunique_numbers_upper",
            "inferred_dtype",
            "dtype_fraction",
            "type_unknown_count",
            "type_null_count",
            "type_fractional_count",
            "type_integral_count",
            "type_boolean_count",
            "type_string_count",
            "nunique_str",
            "nunique_str_lower",
            "nunique_str_upper",
            "quantile_0.0000",
            "quantile_0.0100",
            "quantile_0.0500",
            "quantile_0.2500",
            "quantile_0.5000",
            "quantile_0.7500",
            "quantile_0.9500",
            "quantile_0.9900",
            "quantile_1.0000",
            "null_rate",
        ]

    def test_get_comparison_metric(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        comparison = create_column_summary_comparison(rand_sig, rand_sig2, "A")

        max_metric = get_comparison_metric(comparison, "max")
        assert max_metric.keys() == {"standard", "signature"}
        assert max_metric["signature"] == comparison.loc["max", "signature"]
        assert max_metric["standard"] == comparison.loc["max", "standard"]

    def test_calculate_histogram_intersection(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        histogram_intersection_different = calculate_histogram_intersection(
            rand_sig, mixed_sig, "A"
        )
        histogram_intersection_same = calculate_histogram_intersection(
            rand_sig, rand_sig, "A"
        )
        assert isinstance(histogram_intersection_different, MetricResult)
        assert histogram_intersection_different.metric_name == "Histogram Intersection"
        assert histogram_intersection_different.value < 1
        assert not histogram_intersection_different.passed
        assert histogram_intersection_same.value == 1
        assert histogram_intersection_same.passed

    def test_calculate_category_histogram_intersection(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        mixed_frame_diff = pd.util.testing.makeMixedDataFrame().assign(
            C=["foo1", "foo1", "foo2", "foo3", "foo3"]
        )
        mixed_sig_diff = new_signature(mixed_frame_diff, "project")
        histogram_intersection_different = calculate_category_histogram_intersection(
            mixed_sig_diff, mixed_sig, "C"
        )
        histogram_intersection_same = calculate_category_histogram_intersection(
            mixed_sig, mixed_sig, "C"
        )
        assert isinstance(histogram_intersection_different, MetricResult)
        assert (
            histogram_intersection_different.metric_name
            == "Category Histogram Intersection"
        )
        assert histogram_intersection_different.value < 1
        assert not histogram_intersection_different.passed
        assert histogram_intersection_same.value == 1
        assert histogram_intersection_same.passed

    def test_calculate_mutual_info(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        mutual_info_different = calculate_mutual_info(rand_sig, mixed_sig, "A")
        mutual_info_same = calculate_mutual_info(rand_sig, rand_sig, "A")
        assert isinstance(mutual_info_different, MetricResult)
        assert mutual_info_different.metric_name == "Normalized Mutual Info"
        assert mutual_info_different.value < 1
        assert not mutual_info_different.passed
        assert mutual_info_same.value == 1
        assert mutual_info_same.passed

    def test_calculate_ks_stat(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        ks_different = calculate_ks_stat(rand_sig, mixed_sig, "A")
        ks_same = calculate_ks_stat(rand_sig, rand_sig, "A")
        assert isinstance(ks_different, MetricResult)
        assert ks_different.metric_name == "Kolmogorov-Smirnov"
        assert ks_different.value < 0.1
        assert not ks_different.passed
        assert ks_same.passed
        assert ks_same.value == 1

    def test_calculate_null_rate_discrepancy(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, missing_sig, difnamed_sig = df_signatures
        nrd_signature_more_nulls = calculate_null_rate_discrepancy(
            missing_sig, rand_sig, "D"
        )
        nrd_signature_less_nulls = calculate_null_rate_discrepancy(
            rand_sig, missing_sig, "D"
        )
        nrd_same = calculate_null_rate_discrepancy(missing_sig, missing_sig, "D")
        assert isinstance(nrd_signature_more_nulls, MetricResult)
        assert nrd_signature_more_nulls.metric_name == "Null Rate Discrepancy"
        assert nrd_signature_more_nulls.value > 0.1
        assert not nrd_signature_more_nulls.passed
        assert nrd_signature_less_nulls.value <= 0.1
        assert nrd_signature_less_nulls.passed
        assert nrd_same.passed
        assert nrd_same.value == 0
