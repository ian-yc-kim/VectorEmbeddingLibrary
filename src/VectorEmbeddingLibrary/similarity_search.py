from abc import ABC, abstractmethod
import os
import psycopg2
import numpy as np

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


class SimilaritySearch(ABC):
    """
    Abstract base class for similarity search.

    Methods
    -------
    index_vector(vector, metadata)
        Abstract method to index a vector with associated metadata.

    query_similar(vector, top_k)
        Abstract method to query similar vectors.
    """

    @abstractmethod
    def index_vector(self, vector, metadata):
        pass

    @abstractmethod
    def query_similar(self, vector, top_k):
        pass


class AstraDBSimilaritySearch(SimilaritySearch):
    """
    Concrete implementation of SimilaritySearch using AstraDB.

    Attributes
    ----------
    keyspace : str
        The keyspace in AstraDB.
    table : str
        The table in AstraDB.
    auth_provider : PlainTextAuthProvider
        Authentication provider for AstraDB.
    cluster : Cluster
        Cassandra cluster instance.
    session : Session
        Cassandra session instance.

    Methods
    -------
    index_vector(vector, metadata)
        Indexes the given vector with associated metadata in AstraDB.

    query_similar(vector, top_k)
        Queries the top_k similar vectors from AstraDB.
    """

    def __init__(
        self,
        keyspace,
        table,
        username,
        password,
        host=None,
        port=None,
        secure_connect_bundle=None,
    ):
        """
        Initializes the AstraDBSimilaritySearch with the provided parameters.

        Parameters
        ----------
        keyspace : str
            The keyspace in AstraDB.
        table : str
            The table in AstraDB.
        username : str
            Username for AstraDB authentication.
        password : str
            Password for AstraDB authentication.
        host : str, optional
            Host for AstraDB connection (default is None).
        port : int, optional
            Port for AstraDB connection (default is None).
        secure_connect_bundle : str, optional
            Path to secure connect bundle for AstraDB (default is None).
        """
        self.keyspace = keyspace
        self.table = table
        self.auth_provider = PlainTextAuthProvider(username=username, password=password)
        if secure_connect_bundle and os.path.exists(secure_connect_bundle):
            # Use secure connect bundle if provided
            self.cluster = Cluster(
                cloud={"secure_connect_bundle": secure_connect_bundle},
                auth_provider=self.auth_provider,
            )
        else:
            # Use host and port if secure connect bundle is not provided
            self.cluster = Cluster(
                contact_points=[host], port=port, auth_provider=self.auth_provider
            )
        self.session = self.cluster.connect()

    def index_vector(self, vector, metadata):
        """
        Indexes the given vector with associated metadata in AstraDB.

        Parameters
        ----------
        vector : list
            The vector to be indexed.
        metadata : dict
            Metadata associated with the vector.
        """
        # Data validation
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise ValueError("Vector must be a list of numbers.")
        if not isinstance(metadata, dict) or "id" not in metadata:
            raise ValueError("Metadata must be a dictionary with an 'id' key.")

        # Insert query
        query = f"INSERT INTO {self.keyspace}.{self.table} (id, vector) VALUES (%s, %s)"
        try:
            self.session.execute(query, (metadata["id"], vector))
        except Exception as e:
            raise RuntimeError(f"Failed to index vector: {e}")

    def index_vectors(self, vectors_metadata):
        """
        Indexes a batch of vectors with associated metadata in AstraDB.

        Parameters
        ----------
        vectors_metadata : list of tuples
            Each tuple contains a vector and its associated metadata.
        """
        for vector, metadata in vectors_metadata:
            self.index_vector(vector, metadata)

    def query_similar(self, vector, top_k):
        """
        Queries the top_k similar vectors from AstraDB.

        Parameters
        ----------
        vector : list
            The vector to query similar vectors for.
        top_k : int
            The number of top similar vectors to return.

        Returns
        -------
        list
            A list of tuples containing the id and similarity score of the top_k similar vectors.
        """
        # Data validation
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise ValueError("Vector must be a list of numbers.")
        if len(vector) == 0:
            raise ValueError("Vector must not be empty.")

        # Use AstraDB's built-in similarity search functions
        query = f"SELECT id, vector FROM {self.keyspace}.{self.table} ORDER BY vector ANN OF {vector} LIMIT {top_k}"
        rows = self.session.execute(query)

        # Calculate cosine similarity
        def cosine_similarity(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        results = [(row.id, cosine_similarity(vector, row.vector)) for row in rows]
        # Sort results by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class PostgreSQLSimilaritySearch(SimilaritySearch):
    """
    Concrete implementation of SimilaritySearch using PostgreSQL.

    Attributes
    ----------
    connection : psycopg2.extensions.connection
        The PostgreSQL connection instance.

    Methods
    -------
    index_vector(vector, metadata)
        Indexes the given vector with associated metadata in PostgreSQL.

    query_similar(vector, top_k)
        Queries the top_k similar vectors from PostgreSQL.
    """

    def __init__(self, host, port, database, username, password):
        """
        Initializes the PostgreSQLSimilaritySearch with the provided parameters.

        Parameters
        ----------
        host : str
            Host for PostgreSQL connection.
        port : int
            Port for PostgreSQL connection.
        database : str
            Database name for PostgreSQL connection.
        username : str
            Username for PostgreSQL authentication.
        password : str
            Password for PostgreSQL authentication.
        """
        self.connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )

    def index_vector(self, vector, metadata):
        """
        Indexes the given vector with associated metadata in PostgreSQL.

        Parameters
        ----------
        vector : list
            The vector to be indexed.
        metadata : dict
            Metadata associated with the vector.
        """
        # Data validation
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise ValueError("Vector must be a list of numbers.")
        if not isinstance(metadata, dict) or "id" not in metadata:
            raise ValueError("Metadata must be a dictionary with an 'id' key.")

        # Insert query
        query = "INSERT INTO vectors (id, vector) VALUES (%s, %s)"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (metadata["id"], vector))
                self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise RuntimeError(f"Failed to index vector: {e}")

    def index_vectors(self, vectors_metadata):
        """
        Indexes a batch of vectors with associated metadata in PostgreSQL.

        Parameters
        ----------
        vectors_metadata : list of tuples
            Each tuple contains a vector and its associated metadata.
        """
        for vector, metadata in vectors_metadata:
            self.index_vector(vector, metadata)

    def query_similar(self, vector, top_k):
        """
        Queries the top_k similar vectors from PostgreSQL.

        Parameters
        ----------
        vector : list
            The vector to query similar vectors for.
        top_k : int
            The number of top similar vectors to return.

        Returns
        -------
        list
            A list of tuples containing the id and similarity score of the top_k similar vectors.
        """
        # Data validation
        if not isinstance(vector, list) or not all(
            isinstance(x, (int, float)) for x in vector
        ):
            raise ValueError("Vector must be a list of numbers.")
        if len(vector) == 0:
            raise ValueError("Vector must not be empty.")

        # Query to fetch all vectors
        query = "SELECT id, vector FROM vectors"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except Exception as e:
            raise RuntimeError(f"Failed to query vectors: {e}")

        # Calculate cosine similarity
        def cosine_similarity(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        results = [(row[0], cosine_similarity(vector, row[1])) for row in rows]
        # Sort results by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
