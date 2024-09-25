import pytest
import numpy as np
from similarity_search import AstraDBSimilaritySearch, PostgreSQLSimilaritySearch
import psycopg2


class MockRow:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector


class MockSession:
    def execute(self, query, parameters=None, **kwargs):
        if "INSERT" in query:
            return
        # Mock data for similarity search
        return [MockRow("sample_id", [0.1, 0.2, 0.3])]


class MockCluster:
    def __init__(self, *args, **kwargs):
        pass  # Do nothing

    def connect(self):
        return MockSession()


@pytest.fixture
def mock_cluster(monkeypatch):
    # Patch the Cluster class in the similarity_search module
    monkeypatch.setattr("similarity_search.Cluster", MockCluster)


@pytest.fixture
def similarity_search(mock_cluster):
    return AstraDBSimilaritySearch(
        keyspace="test_keyspace",
        table="test_table",
        username="test_user",
        password="test_pass",
        host="test_host",
        port=9042,
        secure_connect_bundle="test_bundle",  # This won't be used due to the mocking
    )


@pytest.fixture
def postgresql_similarity_search(monkeypatch):
    # Mock the psycopg2 connection and cursor
    class MockCursor:
        def execute(self, query, params=None):
            if "INSERT" in query:
                return
            return [("sample_id", [0.1, 0.2, 0.3])]

        def fetchall(self):
            return [("sample_id", [0.1, 0.2, 0.3])]

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: MockConnection())
    return PostgreSQLSimilaritySearch(
        host="test_host",
        port=5432,
        database="test_db",
        username="test_user",
        password="test_pass"
    )


def test_index_vector(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {"id": "sample_id"}
    similarity_search.index_vector(vector, metadata)
    # If no exception is raised, the test passes


def test_index_vector_invalid_vector(similarity_search):
    vector = "invalid_vector"
    metadata = {"id": "sample_id"}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.index_vector(vector, metadata)


def test_index_vector_invalid_metadata(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = "invalid_metadata"
    with pytest.raises(
        ValueError, match="Metadata must be a dictionary with an 'id' key."
    ):
        similarity_search.index_vector(vector, metadata)


def test_index_vector_missing_id(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {"name": "sample"}  # Missing 'id' key
    with pytest.raises(
        ValueError, match="Metadata must be a dictionary with an 'id' key."
    ):
        similarity_search.index_vector(vector, metadata)


def test_index_vector_invalid_vector_dimensions(similarity_search):
    vector = [0.1, 0.2, "a"]  # Contains a non-numeric value
    metadata = {"id": "sample_id"}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.index_vector(vector, metadata)


def test_index_vectors(similarity_search):
    vectors_metadata = [
        ([0.1, 0.2, 0.3], {"id": "sample_id_1"}),
        ([0.4, 0.5, 0.6], {"id": "sample_id_2"}),
    ]
    similarity_search.index_vectors(vectors_metadata)
    # If no exception is raised, the test passes


def test_index_vector_duplicate_id(postgresql_similarity_search, monkeypatch):
    class MockCursor:
        def execute(self, query, parameters=None):
            if "INSERT" in query:
                raise Exception("Duplicate ID")
            return []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

    monkeypatch.setattr(postgresql_similarity_search, "connection", MockConnection())

    vector = [0.1, 0.2, 0.3]
    metadata = {"id": "sample_id"}
    with pytest.raises(RuntimeError, match="Failed to index vector: Duplicate ID"):
        postgresql_similarity_search.index_vector(vector, metadata)


def test_query_similar(similarity_search):
    vector = [0.1, 0.2, 0.3]
    top_k = 1
    expected_result = [
        ("sample_id", 1.0)
    ]  # Cosine similarity of identical vectors is 1.0
    result = similarity_search.query_similar(vector, top_k)
    assert result == expected_result


def test_query_similar_empty_vector(similarity_search):
    vector = []
    top_k = 1
    with pytest.raises(ValueError, match="Vector must not be empty."):
        similarity_search.query_similar(vector, top_k)


def test_query_similar_invalid_vector(similarity_search):
    vector = "invalid_vector"
    top_k = 1
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.query_similar(vector, top_k)


def test_query_similar_non_numeric_vector(similarity_search):
    vector = [0.1, 0.2, "a"]
    top_k = 1
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.query_similar(vector, top_k)


def test_query_similar_no_results(postgresql_similarity_search, monkeypatch):
    class MockCursor:
        def execute(self, query, parameters=None):
            pass  # Do nothing

        def fetchall(self):
            return []  # No results

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

    monkeypatch.setattr(postgresql_similarity_search, "connection", MockConnection())

    vector = [0.1, 0.2, 0.3]
    top_k = 5
    result = postgresql_similarity_search.query_similar(vector, top_k)
    assert result == []


def test_query_similar_large_dataset(postgresql_similarity_search, monkeypatch):
    import numpy as np
    
    class MockCursor:
        def execute(self, query, parameters=None):
            pass  # Do nothing

        def fetchall(self):
            mock_rows = []
            for i in range(1000):
                id = f"sample_id_{i}"
                vector = [np.random.rand() for _ in range(3)]
                mock_rows.append((id, vector))
            return mock_rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

    monkeypatch.setattr(postgresql_similarity_search, "connection", MockConnection())

    vector = [0.1, 0.2, 0.3]
    top_k = 10
    result = postgresql_similarity_search.query_similar(vector, top_k)
    assert len(result) == top_k
    # Verify that results are sorted by similarity
    similarities = [sim for _, sim in result]
    assert similarities == sorted(similarities, reverse=True)


def test_postgresql_index_vector(postgresql_similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {"id": "sample_id"}
    postgresql_similarity_search.index_vector(vector, metadata)
    # If no exception is raised, the test passes


def test_postgresql_index_vector_invalid_vector(postgresql_similarity_search):
    vector = "invalid_vector"
    metadata = {"id": "sample_id"}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        postgresql_similarity_search.index_vector(vector, metadata)


def test_postgresql_index_vector_invalid_metadata(postgresql_similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = "invalid_metadata"
    with pytest.raises(
        ValueError, match="Metadata must be a dictionary with an 'id' key."
    ):
        postgresql_similarity_search.index_vector(vector, metadata)


def test_postgresql_index_vector_missing_id(postgresql_similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {"name": "sample"}  # Missing 'id' key
    with pytest.raises(
        ValueError, match="Metadata must be a dictionary with an 'id' key."):
        postgresql_similarity_search.index_vector(vector, metadata)


def test_postgresql_index_vector_invalid_vector_dimensions(postgresql_similarity_search):
    vector = [0.1, 0.2, "a"]  # Contains a non-numeric value
    metadata = {"id": "sample_id"}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        postgresql_similarity_search.index_vector(vector, metadata)


def test_postgresql_index_vectors(postgresql_similarity_search):
    vectors_metadata = [
        ([0.1, 0.2, 0.3], {"id": "sample_id_1"}),
        ([0.4, 0.5, 0.6], {"id": "sample_id_2"}),
    ]
    postgresql_similarity_search.index_vectors(vectors_metadata)
    # If no exception is raised, the test passes


def test_postgresql_query_similar(postgresql_similarity_search):
    vector = [0.1, 0.2, 0.3]
    top_k = 1
    expected_result = [
        ("sample_id", 1.0)
    ]  # Cosine similarity of identical vectors is 1.0
    result = postgresql_similarity_search.query_similar(vector, top_k)
    assert result == expected_result


def test_postgresql_query_similar_empty_vector(postgresql_similarity_search):
    vector = []
    top_k = 1
    with pytest.raises(ValueError, match="Vector must not be empty."):
        postgresql_similarity_search.query_similar(vector, top_k)


def test_postgresql_query_similar_invalid_vector(postgresql_similarity_search):
    vector = "invalid_vector"
    top_k = 1
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        postgresql_similarity_search.query_similar(vector, top_k)


def test_postgresql_query_similar_non_numeric_vector(postgresql_similarity_search):
    vector = [0.1, 0.2, "a"]
    top_k = 1
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        postgresql_similarity_search.query_similar(vector, top_k)


def test_postgresql_query_similar_no_results(postgresql_similarity_search, monkeypatch):
    class MockCursor:
        def execute(self, query, parameters=None):
            pass  # Do nothing

        def fetchall(self):
            return []  # No results

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

    monkeypatch.setattr(postgresql_similarity_search, "connection", MockConnection())

    vector = [0.1, 0.2, 0.3]
    top_k = 5
    result = postgresql_similarity_search.query_similar(vector, top_k)
    assert result == []


def test_postgresql_query_similar_large_dataset(postgresql_similarity_search, monkeypatch):
    import numpy as np
    
    class MockCursor:
        def execute(self, query, parameters=None):
            pass  # Do nothing

        def fetchall(self):
            mock_rows = []
            for i in range(1000):
                id = f"sample_id_{i}"
                vector = [np.random.rand() for _ in range(3)]
                mock_rows.append((id, vector))
            return mock_rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockConnection:
        def cursor(self):
            return MockCursor()

    monkeypatch.setattr(postgresql_similarity_search, "connection", MockConnection())

    vector = [0.1, 0.2, 0.3]
    top_k = 10
    result = postgresql_similarity_search.query_similar(vector, top_k)
    assert len(result) == top_k
    # Verify that results are sorted by similarity
    similarities = [sim for _, sim in result]
    assert similarities == sorted(similarities, reverse=True)
