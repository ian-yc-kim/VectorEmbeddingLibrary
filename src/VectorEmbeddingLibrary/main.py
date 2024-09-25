import os
from embedding import OpenAIEmbedder
from similarity_search import AstraDBSimilaritySearch


def main():
    try:
        # Initialize the embedder and similarity search
        api_key = os.getenv('OPENAI_API_KEY')
        embedder = OpenAIEmbedder(api_key)
        
        keyspace = os.getenv('ASTRADB_KEYSPACE')
        table = os.getenv('ASTRADB_TABLE')
        username = os.getenv('ASTRADB_USERNAME')
        password = os.getenv('ASTRADB_PASSWORD')
        secure_connect_bundle = os.getenv('ASTRADB_SECURE_CONNECT_BUNDLE')
        similarity_search = AstraDBSimilaritySearch(keyspace, table, username, password, secure_connect_bundle)

        # Embed a sample text
        sample_text = 'This is a sample text for embedding.'
        vector = embedder.embed_text(sample_text)
        print(f'Embedded vector: {vector}')

        # Index the vector
        metadata = {'id': 'sample_id'}
        similarity_search.index_vector(vector, metadata)
        print('Vector indexed successfully.')

        # Query similar vectors
        top_k = 5
        similar_vectors = similarity_search.query_similar(vector, top_k)
        print(f'Top {top_k} similar vectors: {similar_vectors}')
    except Exception as e:
        print(f'Error: {str(e)}')


if __name__ == '__main__':
    main()
