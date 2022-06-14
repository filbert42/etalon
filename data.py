from mlops_monitoring.signature import Signature
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    BigInteger,
    SmallInteger,
    String,
    LargeBinary,
    DateTime,
)
from fastapi import HTTPException
from sqlalchemy.orm import registry
from sqlalchemy.orm.decl_api import DeclarativeMeta
from abc import ABC, abstractmethod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import urllib.parse
import datetime
import pandas as pd
from typing import Dict, Any, Type
from whylogs.core.datasetprofile import DatasetProfile


class Base(metaclass=DeclarativeMeta):
    __abstract__ = True

    # these are supplied by the sqlalchemy2-stubs, so may be omitted
    # when they are installed
    __table__: Table
    mapper_registry = registry()
    registry = mapper_registry
    metadata = mapper_registry.metadata


class SQLSignature(Base):
    __tablename__ = "stub"
    signature_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    project_name = Column(String, index=True)
    is_standard = Column(SmallInteger)
    signature_binary = Column(LargeBinary)
    upload_date = Column(DateTime, default=True)


class SQLConnection:
    def __init__(self, server_address, table_name):
        self._server_address = server_address
        self._table_name = table_name
        self.SQLSignature = self._get_table()

    @property
    def server_address(self):
        return self._server_address

    @server_address.setter
    def server_address(self, value):
        raise AttributeError(
            "Sorry, you can't change the server address after Reader/Writer initialization"
        )

    @property
    def signatures_table_name(self):
        return self._table_name

    @signatures_table_name.setter
    def signatures_table_name(self, value):
        raise AttributeError(
            "Sorry, you can't change the signatures table name after Reader/Writer initialization"
        )

    def _create_connection_string(self) -> str:
        return (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            + f"SERVER={self.server_address};DATABASE=Mechkar;Trusted_Connection=yes"
        )

    def _create_connection(self):
        params = urllib.parse.quote_plus(string=self._create_connection_string())
        engine = create_engine(
            "mssql+pyodbc:///?odbc_connect=%s" % params,
            connect_args={"check_same_thread": False},
        )
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return Session

    def _get_table(self) -> Type[SQLSignature]:

        table_name_exact = self.signatures_table_name.split(".")[-1]
        schema_name = self.signatures_table_name.split(".")[0]
        table = SQLSignature
        table.__table__.name = table_name_exact
        table.__table__.schema = schema_name
        return table


class Writer(ABC):
    @abstractmethod
    def write_signature(self, signature: Signature) -> None:
        raise NotImplementedError


class Reader(ABC):
    @abstractmethod
    def read_signature(self, signature_id: int) -> Signature:
        raise NotImplementedError

    @abstractmethod
    def read_project_standard(self, project_name: str) -> Signature:
        raise NotImplementedError


class SQLWriter(SQLConnection, Writer):
    def write_signature(self, signature: Signature) -> None:
        data_for_uploading = self._prepare_signature_for_uploading(signature)
        self._write_signature_to_db(data_for_uploading)

    def _write_signature_to_db(self, data_for_uploading: SQLSignature) -> None:
        """
        Helper for writing ready dictionary to the database.

        Args:
            data_for_uploading: dictionary with artifactory table columns as keys.
        """
        con = self._create_connection()
        with con() as session:
            session.add(data_for_uploading)
            session.commit()
            session.refresh(data_for_uploading)

        return None

    def update_standard(self, signature: Signature) -> None:
        """
        Helper for updating standard using ready dictionary to the database.

        Args:
            signature: signature with profile to use as a new standard and a project name of the relevant project.

        """
        data_for_uploading = self._prepare_signature_for_uploading(signature)
        data_for_uploading.is_standard = 1
        con = self._create_connection()
        with con() as session:
            (
                session.query(self.SQLSignature)
                .filter(
                    self.SQLSignature.project_name == data_for_uploading.project_name,
                    self.SQLSignature.is_standard == 1,
                )
                .update({"is_standard": 0})
            )
            session.add(data_for_uploading)
            session.commit()
            session.refresh(data_for_uploading)

    def _prepare_signature_for_uploading(self, signature: Signature) -> SQLSignature:
        proto_signature = signature.profile.to_protobuf().SerializeToString()
        signature_item = self.SQLSignature(
            project_name=signature.project_name,
            is_standard=0,
            signature_binary=proto_signature,
            upload_date=datetime.datetime.now(),
        )

        return signature_item


class SQLReader(SQLConnection, Reader):
    def read_signature(self, signature_id: int) -> Signature:
        raw_signature = self._get_raw_signature_by_id(signature_id)
        signature = self._parse_raw_signarture(raw_signature)
        return signature

    def read_project_standard(self, project_name: str) -> Signature:
        con = self._create_connection()
        with con() as session:
            rawdata = (
                session.query(self.SQLSignature)
                .filter(
                    self.SQLSignature.project_name == project_name,
                    self.SQLSignature.is_standard == 1,
                )
                .first()
            )
            if not rawdata:
                raise HTTPException(
                    status_code=400,
                    detail=f"Standard for project {project_name} not found in the database",
                )
            return self._parse_raw_signarture(rawdata)

    def _get_raw_signature_by_id(self, signature_id: int) -> SQLSignature:
        con = self._create_connection()
        with con() as session:
            rawdata = (
                session.query(self.SQLSignature)
                .filter(self.SQLSignature.signature_id == signature_id)
                .first()
            )

        return rawdata

    def _parse_raw_signarture(self, raw_signature: SQLSignature) -> Signature:
        profile = DatasetProfile.from_protobuf_string(raw_signature.signature_binary)
        return Signature(profile, raw_signature.project_name)


def get_project_standard(project_name: str, reader: Reader) -> Signature:
    return reader.read_project_standard(project_name)
