import os

import gradio as gr

from retriever import Retriever
from reranker import Reranker
from llm_wrapper import LLMWrapper
from utils import load_chunks

# disable tokenizers parallelism to avoid issues with Hugging Face models
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load chunks
chunks = load_chunks()


def answer_question(api_key, query, bm25_enabled, semantic_enabled):
    """
    Answer the user's question using the LLM model.
    """
    if not api_key:
        return "Please enter your LLM API key.", "", ""

    # initialize components
    retriever = Retriever(
        chunks, bm25_enabled=bm25_enabled, semantic_enabled=semantic_enabled
    )
    retrieved_chunks = retriever.retrieve(query)
    reranker = Reranker()
    reranked_chunks = reranker.rerank(query, retrieved_chunks)

    # prepare the context from the top chunks
    context = "\n".join([f"[{idx}] {chunk}" for idx, chunk in reranked_chunks])

    # generate prompt for LLM
    prompt = (
        f"Answer the question based on the context below."
        "ALWAYS cite the sources from context as numbers in square brackets."
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # generate answer
    llm = LLMWrapper(api_key)
    answer = llm.generate_answer(prompt)

    # display the chunks used
    retrieved_docs = "\n".join(
        [f"[{idx}] {chunk[:200]}..." for idx, chunk in retrieved_chunks]
    )
    reranked_docs = "\n".join(
        [f"[{idx}] {chunk[:200]}..." for idx, chunk in reranked_chunks]
    )

    return answer.strip(), retrieved_docs, reranked_docs


# gradio interface
description = """
    # Witcher Book Q&A System
    
    This application answers user questions using the first book of the Witcher series as the knowledge base. It uses a Retrieval-Augmented Generation (RAG) approach, combining keyword search (BM25) and semantic search to find relevant information and generate answers using the LLM model.
    
    **Instructions:**
    
    - Enter your LLM API key.
    - Type your question.
    - Choose whether to enable BM25 keyword search and/or Semantic search.
    - Click 'Submit' to get the answer.
"""

with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        api_key_input = gr.Textbox(label="LLM API Key", type="password")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your question here")
    with gr.Row():
        bm25_checkbox = gr.Checkbox(label="Enable BM25 Keyword Search", value=True)
        semantic_checkbox = gr.Checkbox(label="Enable Semantic Search", value=True)
    submit_button = gr.Button("Submit")
    with gr.Tab("Answer"):
        answer_output = gr.Textbox(label="Answer", lines=5)
    with gr.Tab("Retrieved Documents"):
        retrieved_docs_output = gr.Textbox(
            label="Documents Retrieved by Retriever", lines=10
        )
    with gr.Tab("Reranked Documents"):
        reranked_docs_output = gr.Textbox(label="Documents after Reranking", lines=10)

    submit_button.click(
        fn=answer_question,
        inputs=[api_key_input, query_input, bm25_checkbox, semantic_checkbox],
        outputs=[answer_output, retrieved_docs_output, reranked_docs_output],
    )

# app entry point
if __name__ == "__main__":
    demo.launch()
