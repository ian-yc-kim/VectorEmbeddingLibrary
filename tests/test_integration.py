import pytest
from embedding import OpenAIEmbedder
from similarity_search import AstraDBSimilaritySearch

class MockOpenAI:
    @staticmethod
    def create(input, model):
        return {'data': [{'embedding': [0.1, 0.2, 0.3]}]}

class MockSession:
    def execute(self, query, parameters=None, **kwargs):
        if 'INSERT' in query:
            return
        return [MockRow('sample_id', [0.1, 0.2, 0.3])]

class MockRow:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

class MockCluster:
    def __init__(self, *args, **kwargs):
        pass
    def connect(self):
        return MockSession()

@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr('openai.Embedding', MockOpenAI)

@pytest.fixture
def mock_cluster(monkeypatch):
    monkeypatch.setattr('similarity_search.Cluster', MockCluster)

@pytest.fixture
def embedder():
    return OpenAIEmbedder(api_key='test_key')

@pytest.fixture

def similarity_search(mock_cluster):
    return AstraDBSimilaritySearch(
        keyspace='test_keyspace',
        table='test_table',
        username='test_user',
        password='test_pass',
        host='test_host',
        port=9042,
        secure_connect_bundle='test_bundle'
    )


def test_integration(mock_openai, embedder, similarity_search):
    sample_text = 'This is a sample text for embedding.'
    vector = embedder.embed_text(sample_text)
    assert vector == [0.1, 0.2, 0.3]

    metadata = {'id': 'sample_id'}
    similarity_search.index_vector(vector, metadata)

    top_k = 1
    similar_vectors = similarity_search.query_similar(vector, top_k)
    assert similar_vectors == [('sample_id', 1.0)]
