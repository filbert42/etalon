import numpy as np
import pytest
import os
from fastapi.testclient import TestClient
from mlops_monitoring.server import app
import pandas as pd
from mlops_monitoring.signature import new_signature
from dotenv import load_dotenv


load_dotenv()
os.system(os.environ["DEV_KEYTAB_COMMAND"])


@pytest.fixture(scope="module")
def sql_server():
    return os.environ.get("TEST_SQL_SERVER")


@pytest.fixture(scope="module")
def sql_signature_table():
    return os.environ.get("TEST_SIGNATURES_TABLE")


@pytest.fixture(scope="module")
def signature():
    np.random.seed(42)
    rand_df = pd.util.testing.makeDataFrame()
    sig = new_signature(rand_df, "project")
    return sig


@pytest.fixture(scope="module")
def test_app():
    return TestClient(app)


@pytest.fixture()
def df_signatures(scope="module"):
    np.random.seed(42)
    rand_df = pd.util.testing.makeDataFrame()
    rand_df2 = pd.util.testing.makeDataFrame()
    mixed_df = pd.util.testing.makeMixedDataFrame()
    mixed_df["C"] = mixed_df["C"].astype("category")
    missing_df = pd.util.testing.makeMissingDataframe()
    difnamed_df = rand_df.rename(columns=str.lower)
    return tuple(
        new_signature(df, "test")
        for df in (rand_df, rand_df2, mixed_df, missing_df, difnamed_df)
    )
