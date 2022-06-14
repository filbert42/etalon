import pandas as pd
from mlops_monitoring.client import (
    save_and_compare_signature,
    update_project_standard,
    get_project_standard,
    save_errors_report,
)
from mlops_monitoring.signature import new_signature
import os
import datetime

os.system("kinit -k -t /workspace/michaelle4.keytab michaelle4")


from sklearn_pandas.pipeline import make_transformer_pipeline
import pan_predictor.API.pipeline as pp
from pan_predictor.data.artifactory import MLFlowArtifactory
from pan_predictor.preprocess.preprocess import (
    preprocess_non_train_features,
    unite_transformed_features,
)


def preprocess_all_data(features, populations, transformers):
    preprocessed_data = {
        key: unite_transformed_features(
            preprocess_non_train_features(
                features[key], transformers, populations[key], 15
            )
        )
        for key, value in features.items()
    }
    return preprocessed_data


project_config = {
    "connection_string": "DRIVER={ODBC Driver 17 for SQL Server};\
                          SERVER=mksqlt125\\instance01;DATABASE=Mechkar;Trusted_Connection=yes",
    "project_name": "hospitalization_risk_monitoring_experiments",
    "schema": "dbo",
    "features": [
        "demographics",
        "utilization",
        "diagnoses",
        "hospitalization",
        "markers",
        "chronic_diagnoses",
        "labs",
        "medications",
        "no_shows",
        "clinical_covariates",
    ],
    "history_months": 36,
    "codes_limit_table": "Mechkar.dbo.hospitalization_risk_panp_limit_codes",
    "outcome_column": None,
    "pop_tables": {
        "2019_01": "Mechkar.[clalit\\MichaelLe4].hospitalization_risk_experiments_2019_01",
        "2019_02": "Mechkar.[clalit\\MichaelLe4].hospitalization_risk_experiments_2019_02",
    },
    "produce_features": False,
    "cores": 120,
}

run_id = "cbf2c5bf11f141ef9f68b83c6e0be7b1"


artifactory = MLFlowArtifactory(
    pp._DEFAULT_MLFLOW_SERVER, "hospitalization_risk_prediction_new"
)
transformers = artifactory.load_artifact_item(run_id, "transformers")
populations, features = pp.fetch_populations_and_features(project_config)
preprocessed_data = preprocess_all_data(features, populations, transformers)


print(str(datetime.datetime.now()))
standard = new_signature(preprocessed_data["2019_01"], "hospitalization_risk_demo")
print(str(datetime.datetime.now()))


# update_project_standard(standard, "http://127.0.0.1:8081")
result_same = save_and_compare_signature(standard, "http://slpmkrins-app03:4200")
save_errors_report(result_same)

print(str(datetime.datetime.now()))
signature = new_signature(preprocessed_data["2019_02"], "hospitalization_risk_demo")
print(str(datetime.datetime.now()))

result = save_and_compare_signature(signature, "http://127.0.0.1:8081")
save_errors_report(result)


import numpy as np

strange_data = preprocessed_data["2019_01"].copy()
strange_data["dmg_num_of_children"] = np.nan
strange_data["dmg_age_in_days"] = np.random.rand(len(strange_data))

print(str(datetime.datetime.now()))
signature_strange = new_signature(strange_data, "hospitalization_risk_demo")
print(str(datetime.datetime.now()))
result_strange = save_and_compare_signature(signature_strange, "http://127.0.0.1:8081")
save_errors_report(result_strange)
