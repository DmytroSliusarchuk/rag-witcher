from chunker import Chunker


def load_chunks():
    """
    Load the text chunks from a file or create them if the file does not exist.
    """
    chunker = Chunker()
    chunks_file = "../data/chunks.pkl"
    chunks = chunker.load_chunks(chunks_file)
    if not chunks:
        # read the book text
        with open("../data/books.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunker.chunk_text(text)
        chunker.save_chunks(chunks, chunks_file)
    return chunks
