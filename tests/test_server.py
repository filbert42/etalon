import pytest
from mlops_monitoring.signature import signature_to_dict, json_to_signature
import json
import whylogs as wl


class TestServerAPI:
    def test_save_and_compare_signature(self, test_app, signature):
        json_to_save = json.dumps(signature_to_dict(signature))
        response = test_app.post("/save_and_compare_signature/", data=json_to_save)
        assert response.status_code == 200

    def test_get_project_standard(self, test_app):
        project_name = "project"
        response = test_app.get(f"/get_project_standard/{project_name}")
        sig = json_to_signature(response.text)
        assert response.status_code == 200
        assert sig.project_name == project_name
        assert isinstance(sig.profile, wl.DatasetProfile)
