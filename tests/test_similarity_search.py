import pytest
import numpy as np
from similarity_search import AstraDBSimilaritySearch

class MockRow:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

class MockSession:
    def execute(self, query, parameters=None, **kwargs):
        if 'INSERT' in query:
            return
        # Mock data for similarity search
        return [MockRow('sample_id', [0.1, 0.2, 0.3])]

class MockCluster:
    def __init__(self, *args, **kwargs):
        pass  # Do nothing
    def connect(self):
        return MockSession()

@pytest.fixture
def mock_cluster(monkeypatch):
    # Patch the Cluster class in the similarity_search module
    monkeypatch.setattr('similarity_search.Cluster', MockCluster)

@pytest.fixture
def similarity_search(mock_cluster):
    return AstraDBSimilaritySearch(
        keyspace='test_keyspace',
        table='test_table',
        username='test_user',
        password='test_pass',
        host='test_host',
        port=9042,
        secure_connect_bundle='test_bundle'  # This won't be used due to the mocking
    )

def test_index_vector(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {'id': 'sample_id'}
    similarity_search.index_vector(vector, metadata)
    # If no exception is raised, the test passes

def test_index_vector_invalid_vector(similarity_search):
    vector = 'invalid_vector'
    metadata = {'id': 'sample_id'}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.index_vector(vector, metadata)

def test_index_vector_invalid_metadata(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = 'invalid_metadata'
    with pytest.raises(ValueError, match="Metadata must be a dictionary with an 'id' key."):
        similarity_search.index_vector(vector, metadata)

def test_index_vector_missing_id(similarity_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {'name': 'sample'}  # Missing 'id' key
    with pytest.raises(ValueError, match="Metadata must be a dictionary with an 'id' key."):
        similarity_search.index_vector(vector, metadata)

def test_index_vector_invalid_vector_dimensions(similarity_search):
    vector = [0.1, 0.2, 'a']  # Contains a non-numeric value
    metadata = {'id': 'sample_id'}
    with pytest.raises(ValueError, match="Vector must be a list of numbers."):
        similarity_search.index_vector(vector, metadata)

def test_index_vectors(similarity_search):
    vectors_metadata = [
        ([0.1, 0.2, 0.3], {'id': 'sample_id_1'}),
        ([0.4, 0.5, 0.6], {'id': 'sample_id_2'})
    ]
    similarity_search.index_vectors(vectors_metadata)
    # If no exception is raised, the test passes

def test_index_vector_duplicate_id(similarity_search, monkeypatch):
    def mock_execute(self, query, parameters=None, **kwargs):
        if 'INSERT' in query:
            raise Exception('Duplicate ID')
        # Mock response for other queries
        return []
    # Use functools.partial to bind self if necessary
    monkeypatch.setattr(similarity_search.session, 'execute', mock_execute.__get__(similarity_search.session))
    vector = [0.1, 0.2, 0.3]
    metadata = {'id': 'sample_id'}
    with pytest.raises(RuntimeError, match='Failed to index vector: Duplicate ID'):
        similarity_search.index_vector(vector, metadata)

def test_query_similar(similarity_search):
    vector = [0.1, 0.2, 0.3]
    top_k = 1
    expected_result = [('sample_id', 1.0)]  # Cosine similarity of identical vectors is 1.0
    result = similarity_search.query_similar(vector, top_k)
    assert result == expected_result

def test_query_similar_empty_vector(similarity_search):
    vector = []
    top_k = 1
    with pytest.raises(ValueError, match='Vector must not be empty.'):
        similarity_search.query_similar(vector, top_k)

def test_query_similar_invalid_vector(similarity_search):
    vector = 'invalid_vector'
    top_k = 1
    with pytest.raises(ValueError, match='Vector must be a list of numbers.'):
        similarity_search.query_similar(vector, top_k)

def test_query_similar_non_numeric_vector(similarity_search):
    vector = [0.1, 0.2, 'a']
    top_k = 1
    with pytest.raises(ValueError, match='Vector must be a list of numbers.'):
        similarity_search.query_similar(vector, top_k)

def test_query_similar_no_results(similarity_search, monkeypatch):
    def mock_execute(query, parameters=None, **kwargs):
        return []  # No results
    monkeypatch.setattr(similarity_search.session, 'execute', mock_execute)
    vector = [0.1, 0.2, 0.3]
    top_k = 5
    result = similarity_search.query_similar(vector, top_k)
    assert result == []

def test_query_similar_large_dataset(similarity_search, monkeypatch):
    # Mock a large number of vectors
    def mock_execute(query, parameters=None, **kwargs):
        mock_rows = []
        for i in range(1000):  # Simulate 1000 vectors
            id = f'sample_id_{i}'
            vector = [np.random.rand() for _ in range(3)]
            mock_rows.append(MockRow(id, vector))
        return mock_rows
    monkeypatch.setattr(similarity_search.session, 'execute', mock_execute)
    vector = [0.1, 0.2, 0.3]
    top_k = 10
    result = similarity_search.query_similar(vector, top_k)
    assert len(result) == top_k
    # Optionally, verify that results are sorted by similarity
    similarities = [sim for _, sim in result]
    assert similarities == sorted(similarities, reverse=True)
