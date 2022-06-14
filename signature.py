import json
from typing import Dict, Any, Tuple
from whylogs.util.protobuf import message_to_json
from whylogs.proto import DatasetProfileMessage, ColumnMessage
from google.protobuf.json_format import Parse
from whylogs.core.datasetprofile import DatasetProfile, ColumnProfile
import datetime
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from typing import NamedTuple


class Signature(NamedTuple):
    profile: DatasetProfile
    project_name: str


def new_signature(data: pd.DataFrame, project_name: str) -> Signature:
    timestamp = datetime.datetime.now()
    profile = profile_dataframe_parallel(data, project_name, timestamp, 15)
    signature = Signature(profile, project_name)
    return signature


def signature_to_dict(signature: Signature) -> Dict[str, Any]:
    proto_signature = message_to_json(signature.profile.to_protobuf())
    sign_dict = {"profile": proto_signature, "project_name": signature.project_name}
    return sign_dict


def parse_profile(profile_string: str) -> DatasetProfile:
    return DatasetProfile.from_protobuf(Parse(profile_string, DatasetProfileMessage()))


def parse_column_profile(profile_string: str) -> ColumnProfile:
    return ColumnProfile.from_protobuf(Parse(profile_string, ColumnMessage()))


def json_to_signature(json_sign: str) -> Signature:
    sign_dict = json.loads(json_sign)
    profile = parse_profile(sign_dict["profile"])
    return Signature(profile, sign_dict["project_name"])


def profile_dataframe_parallel(
    data: pd.DataFrame, project_name: str, timestamp: datetime.datetime, cores: int
) -> DatasetProfile:
    profile = DatasetProfile(project_name, timestamp)
    colnames = [str(col) for col in data.columns]
    values = [data[col].values for col in colnames]
    colname_values = zip(colnames, values)
    with Pool(cores) as p:
        column_profiles = p.map(build_column_profile, colname_values)
    profile = DatasetProfile(project_name, timestamp)
    profile.columns = {
        col: parse_column_profile(prof) for col, prof in zip(colnames, column_profiles)
    }
    return profile


def build_column_profile(colname_values: Tuple[str, np.ndarray]) -> ColumnProfile:
    colname, values = colname_values
    profile = ColumnProfile(colname, constraints=None)
    for val in values:
        profile.track(val)
    return message_to_json(profile.to_protobuf())


def get_summary(signature: Signature) -> pd.DataFrame:
    summary_cols = [
        "column",
        "count",
        "numeric_count",
        "type_null_count",
        "type_string_count",
        "max",
        "mean",
        "min",
        "stddev",
        "nunique_numbers",
        "nunique_str",
        "quantile_0.0000",
        "quantile_0.0100",
        "quantile_0.0500",
        "quantile_0.2500",
        "quantile_0.5000",
        "quantile_0.7500",
        "quantile_0.9500",
        "quantile_0.9900",
        "quantile_1.0000",
    ]
    return signature.profile.flat_summary()["summary"].loc[:, summary_cols]
