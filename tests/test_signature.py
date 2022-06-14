import pytest
from mlops_monitoring.signature import *
import json
import whylogs as wl
import pandas as pd
from whylogs.proto import DatasetProfileMessage
from google.protobuf.json_format import Parse


class TestSignature:
    def test_signature_to_dict(self, signature):
        res = json.dumps(signature_to_dict(signature))
        reverse = json.loads(res)
        new = parse_profile(reverse["profile"])
        old_summary = (
            signature.profile.flat_summary()["summary"]
            .sort_values("column")
            .reset_index(drop=True)
        )
        new_summary = (
            new.flat_summary()["summary"].sort_values("column").reset_index(drop=True)
        )

        assert isinstance(res, str)
        assert len(res) != 0
        assert old_summary.equals(new_summary)
        assert reverse["project_name"] == signature.project_name

    def test_json_to_signature(self, signature):
        jsoned = json.dumps(signature_to_dict(signature))
        new_signature = json_to_signature(jsoned)
        old_summary = (
            signature.profile.flat_summary()["summary"]
            .sort_values("column")
            .reset_index(drop=True)
        )
        new_summary = (
            new_signature.profile.flat_summary()["summary"]
            .sort_values("column")
            .reset_index(drop=True)
        )

        assert isinstance(new_signature, Signature)
        assert old_summary.equals(new_summary)
        assert new_signature.project_name == signature.project_name

    def test_new_signature(self, signature):
        assert isinstance(signature, Signature)
        assert isinstance(signature.profile, wl.DatasetProfile)
        assert signature.project_name == "project"

    def test_get_summary(self, signature):
        result = get_summary(signature)
        assert isinstance(result, pd.DataFrame)
        assert {"max", "min", "count", "stddev", "mean", "quantile_0.5000"}.issubset(
            set(result.columns.to_list())
        )
