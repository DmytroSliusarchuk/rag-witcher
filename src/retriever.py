import os
import pickle
from typing import List, Tuple

import numpy as np
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class Retriever:
    """
    Retrieves relevant documents using BM25 and Semantic Search.
    """

    def __init__(
        self,
        chunks: List[str],
        bm25_enabled: bool = True,
        semantic_enabled: bool = True,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        tokenizer_name: str = "bert-base-uncased",
        semantic_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the Retriever.

        :param chunks: List of text chunks.
        :param bm25_enabled: Whether to enable BM25 keyword search.
        :param semantic_enabled: Whether to enable semantic search.
        :param bm25_weight: Weight of BM25 scores when combining.
        :param semantic_weight: Weight of semantic search scores when combining.
        :param tokenizer_name: Name of the tokenizer to use.
        :param semantic_model_name: Name of the semantic model to use.
        """
        self.chunks = chunks
        self.bm25_enabled = bm25_enabled
        self.semantic_enabled = semantic_enabled
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.tokenizer_name = tokenizer_name
        self.semantic_model_name = semantic_model_name

        self.bm25 = None
        self.tokenizer = None
        self.semantic_model = None
        self.chunk_embeddings = None

        if self.bm25_enabled:
            self._init_bm25()

        if self.semantic_enabled:
            self._init_semantic_search()

    def _init_bm25(self):
        """
        Initialize BM25 retriever.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenized_corpus = [
            self.tokenizer.tokenize(chunk.lower()) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _init_semantic_search(self):
        """
        Initialize semantic search retriever.
        """
        self.semantic_model = SentenceTransformer(self.semantic_model_name)
        self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self):
        """
        Load precomputed embeddings or compute them if not available.
        """
        embeddings_file = "../data/embeddings.pkl"
        if os.path.exists(embeddings_file):
            with open(embeddings_file, "rb") as f:
                self.chunk_embeddings = pickle.load(f)
        else:
            self.chunk_embeddings = self.semantic_model.encode(
                self.chunks, convert_to_tensor=True
            )
            with open(embeddings_file, "wb") as f:
                pickle.dump(self.chunk_embeddings, f)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, str]]:
        """
        Retrieve relevant documents for the query.

        :param query: The user's query.
        :param top_k: Number of top documents to retrieve.
        :return: List of tuples containing chunk indices and chunks.
        """
        scores = np.zeros(len(self.chunks))

        if self.bm25_enabled:
            tokenized_query = self.tokenizer.tokenize(query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_scores = np.array(bm25_scores)
            scores += self.bm25_weight * bm25_scores

        if self.semantic_enabled:
            query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
            semantic_scores = torch.nn.functional.cosine_similarity(
                query_embedding, self.chunk_embeddings
            )
            semantic_scores = semantic_scores.cpu().numpy()
            scores += self.semantic_weight * semantic_scores

        # get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved_chunks = [(idx, self.chunks[idx]) for idx in top_indices]

        return retrieved_chunks
