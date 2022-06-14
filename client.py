import requests
import json
from typing import List
from mlops_monitoring.signature import signature_to_dict, json_to_signature, Signature
from mlops_monitoring.compare import ComparingReport


def save_and_compare_signature(
    signature: Signature, server_address: str
) -> ComparingReport:
    """Send signature to the monitoring server to receive a comparing report for signature's standard.

    Args:
        signature: A signature object generated from data that we want to store and compare. Should have correct project name.
        server_address: An address of a monitoring server, including port.

    Returns:
        A ComparingReport object that contains a short status message and a dictionary with failed tests per data column.

    Raises:
        ConnectionError: An error occured during connection to the server.
        HTTPError: Code 400 means that server couldn't find standard for the given signature's project.

    """
    json_to_save = json.dumps(signature_to_dict(signature))
    response = _send_post_request(
        json_to_save, server_address, "/save_and_compare_signature/"
    )
    return ComparingReport(*response.json())


def update_project_standard(new_standard: Signature, server_address: str) -> None:
    """Send singature to the monitoring server and mark it as related project signature.

    Args:
        new_standard: A signature object containing data profile to use as new standard and a project name.

    Raises:
        ConnectionError: An error occured during connection to the server.
    """
    json_to_save = json.dumps(signature_to_dict(new_standard))
    _send_post_request(json_to_save, server_address, "/update_project_standard/")
    return None


def get_project_standard(project_name: str, server_address: str) -> Signature:
    """Get specified project standard from the monitoring server.

    Args:
        project_name: A project name that will be used to find relevant project standard.
        server_address: An address of a monitoring server.

    Raises:
        ConnectionError: An error occured during connection to the server.
        HTTPError: Code 400 means that server couldn't find standard for the given signature's project.
    """
    uri = f"{server_address}/get_project_standard/{project_name}"
    response = requests.get(uri)
    response.raise_for_status()
    standard = json_to_signature(response.text)
    return standard


def save_errors_report(report: ComparingReport) -> None:
    """Save comparing report in the form of a simple txt file.

    Args:
        report: A ComparingReport object to be saved.

    Raises:
        IOError: Something gone wrong during file saving.
    """
    with open(f"{report.project_name}_data_health_report.txt", "w") as file:
        file.write(f"Data Monitoring Report for: {report.project_name} project\n\n")
        file.write(f"Message: {report.message}\n\n")
        if report.failed_columns_stats:
            file.write("Failed tests for columns:\n")
            for k, v in report.failed_columns_stats.items():
                file.write(f"{k}: {v}\n")
            file.write("\n\n\n\n")

        if report.all_columns_stats:
            file.write(f"All columns stats:\n")
            for k, v in report.all_columns_stats.items():
                file.write(f"{k}: {v}\n")

    return None


def _send_post_request(
    data: str, server_address: str, gateway: str
) -> requests.Response:
    # Helper for sending post requests

    uri = f"{server_address}{gateway}"
    response = requests.post(uri, data=data)
    response.raise_for_status()
    return response
