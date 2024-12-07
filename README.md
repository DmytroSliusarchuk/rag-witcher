# Witcher Book Q&A System

This project is a Retrieval-Augmented Generation (RAG) application that answers user queries using the first book of the Witcher series as a knowledge base. It combines traditional keyword-based retrieval (BM25) and semantic search (SentenceTransformer embeddings) to find relevant information from a large text. The selected results are then re-ranked using a cross-encoder model, and finally, a Large Language Model (LLM) provides an answer supported by citations.

## General Components Description

The system follows these main steps:
1. **Data Source**: The first book of the Witcher series is used as the knowledge base.
2. **Chunking**: The source text (the first book of the Witcher series) is split into manageable chunks.
3. **Retriever**: When a user asks a question, the system retrieves relevant chunks using:
   - BM25 keyword-based retrieval.
   - Semantic search using embeddings.
   
   You can enable or disable each retrieval method. If both are active, the final score is a weighted combination (30% BM25, 70% semantic).
   
4. **Reranker**: Retrieved chunks are refined using a cross-encoder to promote the most relevant passages to the top.
5. **LLM**: A Large Language Model (via the Groq API) is prompted with the top re-ranked chunks and the user's question. It generates an answer, citing the sources (chunk indexes) directly in the response.
6. **UI**: Gradio provides a user-friendly interface to interact with the system.
7. **Citations**: The system provides citations for the passages that support the generated answer. These citations are displayed in square brackets [X] in the final answer.

## Detailed Components Description

### `chunker.py`
- **Purpose**: Splits the input text into overlapping chunks to improve retrieval granularity.
- **Work Flow**:
  - The text is split into chunks of a specified size. By default, each chunk contains 1000 characters.
  - Overlapping chunks are created to ensure context continuity. By default, the overlap is 50 characters.
  
### `retriever.py`
- **Purpose**: Finds the most relevant chunks for a given query. By default, BM25 and semantic search are used in combination and 10 chunks are retrieved.
- **Methods**:
  - **BM25 Retrieval**: Keyword-based retrieval using `rank_bm25` library. `BM25Okapi` method is used. Also this component use `AutoTokenizer` from `transformers` library to tokenize and encode the text. The `bert-base-uncased` model is used for encoding.
  - **Semantic Retrieval**: Embedding-based retrieval using `SentenceTransformers`. The `all-MiniLM-L6-v2` model is used for encoding and cosine similarity for scoring.
  - **Combination**: If both methods are enabled, scores are combined (weighted sum). The default weights are 0.3 for BM25 and 0.7 for semantic search.

### `reranker.py`
- **Purpose**: Uses a cross-encoder model to re-rank the retrieved chunks for improved relevance. The `cross-encoder/stsb-distilroberta-base` model from `SentenceTransformers` is used. By default, the top 5 chunks are returned.

### `llm_wrapper.py`
- **Purpose**: Communicates with the `Groq LLM API` to generate a final answer based on the query and retrieved context. The `llama3-8b-8192` model is used.

### `main.py`
- **Purpose**: Provides the Gradio interface to the user, orchestrating the entire pipeline.
- **Features**:
  - Text input for LLM API key and user queries.
  - Checkboxes to enable/disable BM25 and semantic search.
  - Display of retrieved documents before and after re-ranking.
  - Final answer generation and display.
  
### `data` Directory
- **Contains**:
  - `books.txt`: The source text.
  - `chunks.pkl`: Cache of text chunks (generated once and reused).
  - `embeddings.pkl`: Cache of computed embeddings (generated once and reused).

## Requirements

- Python 3.12+
- A Groq API key is required to use the LLM model.

## Data preparation

- Ensure `books.txt` is in the `data/` folder.
- `chunks.pkl` and `embeddings.pkl` should also be in the `data/` folder. If they are not present, they will be generated automatically during the first run.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DmytroSliusarchuk/rag-witcher.git
   cd rag-witcher
    ```

2. Create a virtual environment and activate it
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python main.py
   ```
   
This will start the Gradio interface, allowing you to interact with the system.

## Usage

1. Enter your Groq API key in the designated field.
2. Type your question in the input box.
3. Enable or disable BM25 and semantic search as needed.
4. Click the "Submit" button to see the system's response.
5. The retrieved documents, re-ranked documents, and final answer will be displayed in the interface.

## Notes

- If you change the model or retrieval parameters, consider removing old `embeddings.pkl` so the system can recompute embeddings.
- Disabling BM25 or semantic search can help you understand how different retrieval methods affect results.
- The citations in square brackets [X] correspond to chunk indices. Scroll through the retrieved documents tabs to see the cited passages.