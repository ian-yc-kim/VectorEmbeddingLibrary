import pytest
from embedding import OpenAIEmbedder
from similarity_search import AstraDBSimilaritySearch, PostgreSQLSimilaritySearch
import psycopg2


class MockOpenAI:
    @staticmethod
    def create(input, model):
        embeddings_dict = {
            "Sample text 1.": [0.1, 0.2, 0.3],
            "Sample text 2.": [0.3, 0.2, 0.1],
            "This is a sample text for embedding.": [0.1, 0.2, 0.3],
        }
        return {"data": [{"embedding": embeddings_dict.get(input, [0.1, 0.2, 0.3])}]}


class MockConnection:
    def __init__(self):
        self.data = {}  # Initialize an empty dictionary to store data

    def cursor(self):
        return MockCursor(self.data)  # Pass the data dictionary to the cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


class MockCursor:
    def __init__(self, data):
        self.data = data  # Use the shared data dictionary
        self.last_query_results = []

    def execute(self, query, params=None):
        if "INSERT" in query:
            # Assuming params are (id, vector)
            self.data[params[0]] = params[1]
        elif "SELECT" in query:
            # Simulate fetching all stored data
            self.last_query_results = [(id, vector) for id, vector in self.data.items()]
        else:
            pass

    def fetchall(self):
        return self.last_query_results

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr("openai.Embedding", MockOpenAI)


@pytest.fixture
def embedder():
    return OpenAIEmbedder(api_key="test_key")


@pytest.fixture
def postgresql_similarity_search(monkeypatch):
    # Adjusted to use the updated MockConnection
    monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: MockConnection())
    return PostgreSQLSimilaritySearch(
        host="test_host",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_pass"
    )


def test_postgresql_batch_indexing(mock_openai, embedder, postgresql_similarity_search):
    sample_texts = ["Sample text 1.", "Sample text 2."]
    vectors = [embedder.embed_text(text) for text in sample_texts]
    metadata_list = [{"id": f"sample_id_{i}"} for i in range(len(vectors))]
    vectors_metadata = list(zip(vectors, metadata_list))

    postgresql_similarity_search.index_vectors(vectors_metadata)

    for vector, metadata in vectors_metadata:
        similar_vectors = postgresql_similarity_search.query_similar(vector, top_k=1)
        assert similar_vectors[0][0] == metadata["id"]
