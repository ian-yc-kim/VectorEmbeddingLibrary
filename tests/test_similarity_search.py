import pytest
from similarity_search import AstraDBSimilaritySearch

class MockSession:
    def execute(self, query, params=None):
        if 'INSERT' in query:
            return
        # Mock data for similarity search
        class MockRow:
            id = 'sample_id'
            vector = [0.1, 0.2, 0.3]
        return [MockRow()]

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

def test_query_similar(similarity_search):
    vector = [0.1, 0.2, 0.3]
    top_k = 1
    expected_result = [('sample_id', 0.0)]
    result = similarity_search.query_similar(vector, top_k)
    assert result == expected_result
