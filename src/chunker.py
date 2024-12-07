import os
import pickle
from typing import List


class Chunker:
    """
    Splits documents into smaller chunks for processing.
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 50):
        """
        Initialize the Chunker.

        :param chunk_size: Number of characters in each chunk.
        :param overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.

        :param text: The input text to be chunked.
        :return: A list of text chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks

    def save_chunks(self, chunks: List[str], filename: str):
        """
        Save the chunks to a pickle file.

        :param chunks: The list of text chunks.
        :param filename: The filename to save the chunks.
        """
        with open(filename, "wb") as f:
            pickle.dump(chunks, f)

    def load_chunks(self, filename: str) -> List[str]:
        """
        Load chunks from a pickle file.

        :param filename: The filename to load the chunks from.
        :return: A list of text chunks.
        """
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                chunks = pickle.load(f)
            return chunks
        else:
            return []
