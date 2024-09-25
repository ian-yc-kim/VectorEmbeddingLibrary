from abc import ABC, abstractmethod
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os

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
    def __init__(self, keyspace, table, username, password, host=None, port=None, secure_connect_bundle=None):
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
            self.cluster = Cluster(cloud={'secure_connect_bundle': secure_connect_bundle}, auth_provider=self.auth_provider)
        else:
            # Use host and port if secure connect bundle is not provided
            self.cluster = Cluster(contact_points=[host], port=port, auth_provider=self.auth_provider)
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
        query = f"INSERT INTO {self.keyspace}.{self.table} (id, vector) VALUES (%s, %s)"
        self.session.execute(query, (metadata['id'], vector))

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
            A list of tuples containing the id and distance of the top_k similar vectors.
        """
        query = f"SELECT id, vector FROM {self.keyspace}.{self.table}"
        rows = self.session.execute(query)
        # Implement a simple similarity search (e.g., Euclidean distance)
        results = []
        for row in rows:
            distance = sum((a - b) ** 2 for a, b in zip(vector, row.vector)) ** 0.5
            results.append((row.id, distance))
        results.sort(key=lambda x: x[1])
        return results[:top_k]
