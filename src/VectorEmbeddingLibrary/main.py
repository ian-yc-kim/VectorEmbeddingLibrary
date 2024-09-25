from config import Config
from embedding import OpenAIEmbedder
from similarity_search import AstraDBSimilaritySearch


def main():
    """
    Main function to run the vector embedding and similarity search process.

    This function loads the configuration, initializes the embedder and similarity search,
    embeds a sample text, indexes the vector, and queries similar vectors.
    """
    try:
        # Load configuration
        config = Config()

        # Initialize the embedder and similarity search
        embedder = OpenAIEmbedder(config.openai_api_key)
        similarity_search = AstraDBSimilaritySearch(
            config.astradb['keyspace'],
            config.astradb['table'],
            config.astradb['username'],
            config.astradb['password'],
            config.astradb['host'],
            config.astradb['port'],
            config.astradb['secure_connect_bundle']
        )

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
