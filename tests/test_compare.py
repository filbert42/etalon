import pytest
import pandas as pd
import numpy as np
from mlops_monitoring.compare import *


class TestCompare:
    def test_compare_signatures_different_columns(self, df_signatures):
        rand_sig, _, _, _, difnamed_sig = df_signatures

        assert (
            compare_signatures(rand_sig, difnamed_sig).message
            == "Error: columns in the signature are not the same as in the standard!"
        )

    def test_collect_failed_metrics(self, df_signatures):
        rand_sig, _, mixed_sig, _, _ = df_signatures
        metrics = [
            calculate_mutual_info,
            calculate_ks_stat,
            calculate_histogram_intersection,
        ]
        columnstats_same = calculate_column_stats(rand_sig, rand_sig, "A", metrics)
        result_same = collect_failed_metrics(columnstats_same)
        assert result_same is None

        columnstats_different = calculate_column_stats(
            rand_sig, mixed_sig, "A", metrics
        )
        result_different = collect_failed_metrics(columnstats_different)
        assert result_different is not None

    def test_compare_signatures_basic(self, df_signatures):
        rand_sig, _, mixed_sig, _, _ = df_signatures
        result = compare_signatures(rand_sig, rand_sig)
        assert isinstance(result, ComparingReport)
        assert result.message == "All fine!"
        assert result.failed_columns_stats is None
        assert result.all_columns_stats is not None

        result_diff = compare_signatures(mixed_sig, rand_sig)
        assert isinstance(result_diff, ComparingReport)
        assert result_diff.message == "Some columns are not OK!"
        assert result_diff.failed_columns_stats is not None
        assert result_diff.all_columns_stats is not None
        assert len(result_diff.all_columns_stats) >= len(result_diff.failed_columns_stats)

    def test_check_same_columns(self, df_signatures):
        rand_sig, rand_sig2, _, _, difnamed_sig = df_signatures
        assert check_same_columns(rand_sig, rand_sig2)
        assert not check_same_columns(rand_sig, difnamed_sig)

    def test_calculate_stats(self, df_signatures):
        rand_sig, rand_sig2, _, _, _ = df_signatures
        rand_sig_stats = calculate_stats(rand_sig, rand_sig2)

        assert set(rand_sig_stats.keys()) == set(rand_sig.profile.columns.keys())

    def test_calculate_numeric_stats(self, df_signatures):
        rand_sig, rand_sig2, mixed_sig, _, _ = df_signatures
        rand_sig_stats = calculate_numeric_stats(rand_sig, rand_sig2)

        assert set(rand_sig_stats.keys()) == set(rand_sig.profile.columns.keys())

        mixed_sig_stats = calculate_numeric_stats(rand_sig, mixed_sig)
        assert set(mixed_sig_stats.keys()) == {"A", "B"}

    def test_calculate_categorical_stats(self, df_signatures):
        _, _, mixed_sig, _, _ = df_signatures

        mixed_sig_stats = calculate_categorical_stats(mixed_sig, mixed_sig)
        assert set(mixed_sig_stats.keys()) == {"C"}

    def test_calculate_column_stats(self, df_signatures):
        rand_sig, rand_sig2, _, _, _ = df_signatures
        metrics = [
            calculate_mutual_info,
            calculate_ks_stat,
            calculate_histogram_intersection,
        ]
        rand_sig_stats = calculate_column_stats(rand_sig, rand_sig2, "A", metrics)

        assert rand_sig_stats[1] == calculate_ks_stat(rand_sig, rand_sig2, "A")
        assert rand_sig_stats[2] == calculate_histogram_intersection(
            rand_sig, rand_sig2, "A"
        )
        assert rand_sig_stats[0] == calculate_mutual_info(rand_sig, rand_sig2, "A")

    def test_get_signature_cols(self, df_signatures):
        rand_sig, _, _, _, difnamed_sig = df_signatures

        assert get_signature_cols(rand_sig) == {"A", "B", "C", "D"}
        assert get_signature_cols(difnamed_sig) == {"a", "b", "c", "d"}

    def test_get_numeric_cols(self, df_signatures):
        rand_sig, _, mixed_sig, _, _ = df_signatures

        assert get_numeric_cols(rand_sig) == {"A", "B", "C", "D"}
        assert get_numeric_cols(mixed_sig) == {"A", "B"}

    def test_get_categorical_cols(self, df_signatures):
        rand_sig, _, mixed_sig, _, _ = df_signatures

        assert get_categorical_cols(rand_sig) == set()
        assert get_categorical_cols(mixed_sig) == {"C"}
