import pytest
from mlops_monitoring.data import SQLWriter, SQLReader
import pandas as pd


class TestSQLWriter:
    @pytest.fixture
    def sql_writer(self, sql_server, sql_signature_table):
        return SQLWriter(sql_server, sql_signature_table)

    def test_sql_writer_init(self, sql_server, sql_signature_table):
        sql_writer = SQLWriter(sql_server, sql_signature_table)
        assert sql_writer.server_address == sql_server
        assert sql_writer.signatures_table_name == sql_signature_table

    def test_sql_write_signature(self, signature, sql_writer):
        # simple test that writting doesn't raise exceptions
        sql_writer.write_signature(signature)


class TestSQLReader:
    @pytest.fixture
    def sql_reader(self, sql_server, sql_signature_table):
        return SQLReader(sql_server, sql_signature_table)

    def test_sql_reader_init(self, sql_server, sql_signature_table):
        sql_reader = SQLReader(sql_server, sql_signature_table)
        assert sql_reader.server_address == sql_server
        assert sql_reader.signatures_table_name == sql_signature_table

    def test_sql_read_signature(self, signature, sql_reader):
        # simple test that writting doesn't raise exceptions
        sig = sql_reader.read_signature(2)
        assert sig.project_name == "project"
        assert sig.profile is not None

    def test_sql_read_project_standard(self, sql_reader):
        result = sql_reader.read_project_standard("project")
        assert result.project_name == "project"
