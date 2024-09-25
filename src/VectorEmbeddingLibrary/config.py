import os
import yaml


class Config:
    """
    Configuration class to handle loading and accessing configuration settings.

    Attributes
    ----------
    config : dict
        Dictionary to store configuration settings.
    """

    def __init__(self, config_file="config.yaml"):
        """
        Initializes the Config class by loading settings from a YAML file and environment variables.

        Parameters
        ----------
        config_file : str, optional
            Path to the YAML configuration file (default is 'config.yaml').
        """
        with open(config_file, "r") as file:
            self.config = yaml.safe_load(file)

        self.config["embedding_service"] = os.getenv(
            "EMBEDDING_SERVICE", self.config.get("embedding_service")
        )
        self.config["database"] = {
            "host": os.getenv("DB_HOST", self.config["database"].get("host")),
            "port": os.getenv("DB_PORT", self.config["database"].get("port")),
            "keyspace": os.getenv(
                "DB_KEYSPACE", self.config["database"].get("keyspace")
            ),
        }
        self.config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        self.config["astradb"] = {
            "keyspace": os.getenv("ASTRADB_KEYSPACE"),
            "table": os.getenv("ASTRADB_TABLE"),
            "username": os.getenv("ASTRADB_USERNAME"),
            "password": os.getenv("ASTRADB_PASSWORD"),
            "secure_connect_bundle": os.getenv("ASTRADB_SECURE_CONNECT_BUNDLE"),
        }
        self.config["postgresql"] = {
            "host": os.getenv("POSTGRESQL_HOST", self.config.get("postgresql", {}).get("host")),
            "port": os.getenv("POSTGRESQL_PORT", self.config.get("postgresql", {}).get("port")),
            "database": os.getenv("POSTGRESQL_DATABASE", self.config.get("postgresql", {}).get("database")),
            "username": os.getenv("POSTGRESQL_USERNAME", self.config.get("postgresql", {}).get("username")),
            "password": os.getenv("POSTGRESQL_PASSWORD", self.config.get("postgresql", {}).get("password")),
        }

    @property
    def embedding_service(self):
        """
        Returns the embedding service configuration.

        Returns
        -------
        str
            The embedding service to use.
        """
        return self.config["embedding_service"]

    @property
    def database(self):
        """
        Returns the database configuration.

        Returns
        -------
        dict
            The database configuration settings.
        """
        return self.config["database"]

    @property
    def openai_api_key(self):
        """
        Returns the OpenAI API key.

        Returns
        -------
        str
            The OpenAI API key.
        """
        return self.config["openai_api_key"]

    @property
    def astradb(self):
        """
        Returns the AstraDB configuration.

        Returns
        -------
        dict
            The AstraDB configuration settings.
        """
        return self.config["astradb"]

    @property
    def postgresql(self):
        """
        Returns the PostgreSQL configuration.

        Returns
        -------
        dict
            The PostgreSQL configuration settings.
        """
        return self.config["postgresql"]
