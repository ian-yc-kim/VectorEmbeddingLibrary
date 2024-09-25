import os
import pytest
from config import Config


class TestConfig:
    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_SERVICE", "openai")
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "9042")
        monkeypatch.setenv("DB_KEYSPACE", "VectorEmbeddingDB")
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_api_key")
        monkeypatch.setenv("ASTRADB_KEYSPACE", "test_keyspace")
        monkeypatch.setenv("ASTRADB_TABLE", "test_table")
        monkeypatch.setenv("ASTRADB_USERNAME", "test_username")
        monkeypatch.setenv("ASTRADB_PASSWORD", "test_password")
        monkeypatch.setenv(
            "ASTRADB_SECURE_CONNECT_BUNDLE", "test_secure_connect_bundle"
        )

    @pytest.fixture
    def config_file(self, tmpdir):
        config_yaml = tmpdir.join("config.yaml")
        config_yaml.write(
            """
embedding_service: 'openai'
database:
  host: 'localhost'
  port: 9042
  keyspace: 'VectorEmbeddingDB'
"""
        )
        return config_yaml

    def test_config_loading(self, config_file):
        config = Config(config_file=str(config_file))
        assert config.embedding_service == "openai"
        assert config.database["host"] == "localhost"
        assert config.database["port"] == "9042"  # Updated to string comparison
        assert config.database["keyspace"] == "VectorEmbeddingDB"
        assert config.openai_api_key == "test_openai_api_key"
        assert config.astradb["keyspace"] == "test_keyspace"
        assert config.astradb["table"] == "test_table"
        assert config.astradb["username"] == "test_username"
        assert config.astradb["password"] == "test_password"
        assert config.astradb["secure_connect_bundle"] == "test_secure_connect_bundle"
