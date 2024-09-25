from abc import ABC, abstractmethod
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

class SimilaritySearch(ABC):
    @abstractmethod
    def index_vector(self, vector, metadata):
        pass

    @abstractmethod
    def query_similar(self, vector, top_k):
        pass

class AstraDBSimilaritySearch(SimilaritySearch):
    def __init__(self, keyspace, table, username, password, secure_connect_bundle):
        self.keyspace = keyspace
        self.table = table
        self.auth_provider = PlainTextAuthProvider(username=username, password=password)
        self.cluster = Cluster(cloud={'secure_connect_bundle': secure_connect_bundle}, auth_provider=self.auth_provider)
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
