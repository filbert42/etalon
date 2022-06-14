from mlops_monitoring.signature import Signature
from mlops_monitoring.metrics import (
    MetricResult,
    calculate_histogram_intersection,
    calculate_ks_stat,
    calculate_mutual_info,
    calculate_null_rate_discrepancy,
    calculate_category_histogram_intersection,
)

from typing import Dict, Any, Set, Tuple, List, Optional, Callable, NewType, Sequence
import numpy as np
import pandas as pd

from typing import NamedTuple


class ComparingReport(NamedTuple):
    project_name: str
    message: str
    all_columns_stats: Optional[Dict[str, str]]
    failed_columns_stats: Optional[Dict[str, str]]


def compare_signatures(signature: Signature, standard: Signature) -> ComparingReport:
    """Compare two signatures and produce comparing report that contains info about failed tests.

    Args:
        signature: A Signature object, by convention contains profile of the new data
        standard: Another Signature object, by convention contains profile with project standard

    Returns:
        A ComparingReport object that contains a short status message and a dictionary with failed tests per data column.
    """
    same_columns = check_same_columns(signature, standard)
    if not same_columns:

        return ComparingReport(
            project_name=signature.project_name,
            message="Error: columns in the signature are not the same as in the standard!",
            all_columns_stats=None,
            failed_columns_stats=None,
        )

    column_stats = calculate_stats(signature, standard)
    report = create_report(signature.project_name, column_stats)
    return report


def create_report(
    project_name: str, columns_stats: Dict[str, List[MetricResult]]
) -> ComparingReport:
    """Create comparing report using calculated metrics.

    Args:
        project_name: A project name to be saved in the report.
        column_stats: Calculated comparison metrics for each column in the signature and standard.

    Returns:
        A ComparingReport object that contains a short status message and a dictionary with failed tests per data column.
    """
    all_columns_report = {
        col: metrics_to_string(metric) for col, metric in columns_stats.items()
    }

    failed_columns_report = {
        colname: res
        for colname, metrics in columns_stats.items()
        if (res := collect_failed_metrics(metrics)) is not None
    }

    if len(failed_columns_report) == 0:
        return ComparingReport(
            project_name,
            message="All fine!",
            failed_columns_stats=None,
            all_columns_stats=all_columns_report,
        )

    return ComparingReport(
        project_name,
        message="Some columns are not OK!",
        failed_columns_stats=failed_columns_report,
        all_columns_stats=all_columns_report,
    )


def collect_failed_metrics(metrics: List[MetricResult]) -> Optional[str]:
    """Collect failed tests from the list of metrics and represent them with a string.

    Args:
        metrics: A list of comparison metric results.

    Returns:
        A string with all failed metrics or None if all metrics are passed.
    """
    failed = [metric for metric in metrics if not metric.passed]
    msg = metrics_to_string(failed)
    return None if len(failed) == 0 else msg


def metrics_to_string(metrics: List[MetricResult]) -> str:
    return ", ".join(
        [f"{metric.metric_name} ({round(metric.value, 5)})" for metric in metrics]
    )


def check_same_columns(signature: Signature, standard: Signature) -> bool:
    return get_signature_cols(signature) == get_signature_cols(standard)


def calculate_stats(
    signature: Signature, standard: Signature
) -> Dict[str, List[MetricResult]]:
    """Calculate comparison metrics for both numeric and non-numeric columns.

    Args: See compare_signatures()

    Returns:
        A dictionary with column names as keys and calculated metrics as values.
    """
    numeric_stats = calculate_numeric_stats(signature, standard)
    non_numeric_stats = calculate_categorical_stats(signature, standard)
    return {**numeric_stats, **non_numeric_stats}


def calculate_numeric_stats(
    signature: Signature, standard: Signature
) -> Dict[str, List[MetricResult]]:
    """Calculate metrics for all numeric columns.

    Args: See compare_signatures()

    Returns:
        A dictionary with column names as keys and calculated metrics as values.
    """
    numeric_cols = get_numeric_cols(standard)
    numeric_stats_functions = [
        calculate_histogram_intersection,
        # calculate_mutual_info,
        # calculate_ks_stat,
        calculate_null_rate_discrepancy,
    ]
    numeric_stats = {
        colname: calculate_column_stats(
            signature, standard, colname, numeric_stats_functions
        )
        for colname in numeric_cols
    }
    return numeric_stats


def calculate_categorical_stats(
    signature: Signature, standard: Signature
) -> Dict[str, List[MetricResult]]:
    """Calculate metrics for all categorical columns.

    Args: See compare_signatures()

    Returns:
        A dictionary with column names as keys and calculated metrics as values.
    """
    categorical_cols = get_categorical_cols(standard)
    stats_functions = [
        calculate_category_histogram_intersection,
    ]
    categorical_stats = {
        colname: calculate_column_stats(signature, standard, colname, stats_functions)
        for colname in categorical_cols
    }
    return categorical_stats


def calculate_column_stats(
    signature: Signature,
    standard: Signature,
    colname: str,
    metrics: Sequence[Callable[[Signature, Signature, str], MetricResult]],
) -> List[MetricResult]:
    """Apply all given metric functions to compare column between to signatures.

    Args:
        See See compare_signatures()
        colname: Column name to compare between signatures.
        metrics: list of metric functions to use for comparison.

    Returns:
        A list of metric results.
    """
    return [metric_func(signature, standard, colname) for metric_func in metrics]


def get_signature_cols(signature: Signature) -> Set[str]:
    return set(signature.profile.columns.keys())


def get_numeric_cols(signature: Signature) -> Set[str]:
    signature_cols = get_signature_cols(signature)
    return {
        col
        for col in signature_cols
        if signature.profile.columns[col].number_tracker.to_summary() is not None
    }


def get_categorical_cols(signature: Signature) -> Set[str]:
    return set(signature.profile.flat_summary()["frequent_strings"].keys())
