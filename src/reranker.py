from typing import List, Tuple

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    """

    def __init__(self, model_name: str = "cross-encoder/stsb-distilroberta-base"):
        """
        Initialize the Reranker.

        :param model_name: The name of the cross-encoder model.
        """
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(
        self, query: str, retrieved_chunks: List[Tuple[int, str]], top_k: int = 5
    ) -> List[Tuple[int, str]]:
        """
        Rerank the retrieved documents.

        :param query: The user's query.
        :param retrieved_chunks: List of tuples containing chunk indices and chunks.
        :param top_k: Number of top documents to return after reranking.
        :return: Reranked list of tuples containing chunk indices and chunks.
        """
        cross_inp = [[query, chunk[1]] for chunk in retrieved_chunks]
        scores = self.cross_encoder.predict(cross_inp)

        # combine the chunks with their scores
        scored_chunks = list(zip(scores, retrieved_chunks))

        # sort by score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # return top_k chunks
        reranked_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
        return reranked_chunks
