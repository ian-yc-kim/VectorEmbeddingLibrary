from abc import ABC, abstractmethod
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os

class SimilaritySearch(ABC):
    @abstractmethod
    def index_vector(self, vector, metadata):
        pass

    @abstractmethod
    def query_similar(self, vector, top_k):
        pass

class AstraDBSimilaritySearch(SimilaritySearch):
    def __init__(self, keyspace, table, username, password, host=None, port=None, secure_connect_bundle=None):
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
        query = f"INSERT INTO {self.keyspace}.{self.table} (id, vector) VALUES (%s, %s)"
        self.session.execute(query, (metadata['id'], vector))

    def query_similar(self, vector, top_k):
        query = f"SELECT id, vector FROM {self.keyspace}.{self.table}"
        rows = self.session.execute(query)
        # Implement a simple similarity search (e.g., Euclidean distance)
        results = []
        for row in rows:
            distance = sum((a - b) ** 2 for a, b in zip(vector, row.vector)) ** 0.5
            results.append((row.id, distance))
        results.sort(key=lambda x: x[1])
        return results[:top_k]
