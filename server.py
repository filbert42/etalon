from fastapi import FastAPI
from pydantic import BaseModel
from mlops_monitoring.signature import Signature, parse_profile, signature_to_dict
from mlops_monitoring.data import SQLWriter, SQLReader, get_project_standard
from mlops_monitoring.compare import compare_signatures
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

SQL_SERVER = os.environ["SQL_SERVER"]
SIGNATURES_TABLE = os.environ["SIGNATURES_TABLE"]


class SignatureMessage(BaseModel):
    profile: str
    project_name: str


app = FastAPI()


def _parse_message(msg: SignatureMessage) -> Signature:
    # Helper to parse json messages
    profile = parse_profile(msg.profile)
    return Signature(profile, msg.project_name)


@app.post("/save_and_compare_signature/")
def save_and_compare_signature(msg: SignatureMessage):
    os.system(os.environ["DEV_KEYTAB_COMMAND"])
    signature = _parse_message(msg)
    reader = SQLReader(SQL_SERVER, SIGNATURES_TABLE)
    standard = get_project_standard(signature.project_name, reader)
    SQLWriter(SQL_SERVER, SIGNATURES_TABLE).write_signature(signature)
    result = compare_signatures(signature, standard)
    return result


@app.post("/update_project_standard/")
def update_project_standard(msg: SignatureMessage):
    os.system(os.environ["DEV_KEYTAB_COMMAND"])
    new_standard = _parse_message(msg)
    SQLWriter(SQL_SERVER, SIGNATURES_TABLE).update_standard(new_standard)


@app.get("/get_project_standard/{project_name}")
def project_standard(project_name: str):
    os.system(os.environ["DEV_KEYTAB_COMMAND"])
    reader = SQLReader(SQL_SERVER, SIGNATURES_TABLE)
    standard = get_project_standard(project_name, reader)
    jsoned_standard = signature_to_dict(standard)
    return jsoned_standard


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4200)
