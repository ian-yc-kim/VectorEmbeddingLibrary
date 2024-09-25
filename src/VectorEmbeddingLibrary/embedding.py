from abc import ABC, abstractmethod
import openai
import logging


class VectorEmbedder(ABC):
    """
    Abstract base class for vector embedding.

    Methods
    -------
    embed_text(text: str) -> list
        Abstract method to embed text into a vector.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list:
        pass


class OpenAIEmbedder(VectorEmbedder):
    """
    Concrete implementation of VectorEmbedder using OpenAI's embedding service.

    Attributes
    ----------
    api_key : str
        API key for accessing OpenAI's services.

    Methods
    -------
    embed_text(text: str) -> list
        Embeds the given text into a vector using OpenAI's embedding model.
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAIEmbedder with the provided API key.

        Parameters
        ----------
        api_key : str
            API key for accessing OpenAI's services.
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def embed_text(self, text: str) -> list:
        """
        Embeds the given text into a vector using OpenAI's embedding model.

        Parameters
        ----------
        text : str
            The text to be embedded.

        Returns
        -------
        list
            The embedded vector.
        """
        try:
            response = openai.Embedding.create(
                input=text, model="text-embedding-ada-002"
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            return []
