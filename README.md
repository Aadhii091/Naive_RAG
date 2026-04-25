# 🧠 Naive RAG Chatbot (Streamlit + Local LLM)

## Overview

This project is a Retrieval-Augmented Generation (RAG) application built with a **Streamlit interface** and a fully **local AI stack**. It allows users to query documents conversationally, with responses grounded in retrieved context.

The system combines semantic retrieval with a lightweight LLM pipeline using **Ollama (Gemma 3:4B)** and **EmbeddingGemma**, enabling private, offline-capable inference.

---

## Why This Project Exists

LLMs alone hallucinate. RAG fixes that by injecting relevant context — but most implementations are either overengineered or API-dependent.

This project focuses on:

* **Local-first architecture** (no external APIs)
* **Transparent retrieval pipeline**
* **Experimentation with chunking strategies**
* **Conversation-aware querying**

---

## What It Does

* Upload and index documents
* Perform semantic search over content
* Generate context-aware answers using an LLM
* Maintain conversational memory
* Experiment with multiple chunking strategies

---

## System Architecture

### High-Level Flow

1. User query enters via Streamlit UI
2. Query is reformulated using conversation history
3. Embeddings generated using EmbeddingGemma
4. Relevant chunks retrieved from ChromaDB (MMR)
5. Context injected into LLM (Gemma 3:4B via Ollama)
6. Response generated and returned to UI

---

## Tech Stack

### Interface

* Streamlit

### Core Backend

* LangChain
* ChromaDB (vector store)

### Models

* Ollama (Gemma 3:4B)
* EmbeddingGemma (for embeddings)

---

## Key Features

* **History-aware conversation** (context retention)
* **Semantic chunking** for better context preservation
* **MMR (Max Marginal Relevance)** for diverse retrieval
* **RecursiveCharacterTextSplitter** for baseline chunking
* **Agentic chunking (experimental)**

---

## Retrieval Pipeline

### Chunking Strategies

1. **RecursiveCharacterTextSplitter**

   * Baseline method
   * Splits text by hierarchy (paragraph → sentence → word)

2. **Semantic Chunking**

   * Groups text based on meaning
   * Preserves context better than fixed splits

3. **Agentic Chunking (Experimental)**

   * Attempts intelligent segmentation using LLM reasoning
   * Still under evaluation

---

## Search Strategy

* Uses **MMR (Max Marginal Relevance)**
* Balances:

  * Relevance to query
  * Diversity of retrieved chunks

---

## Installation

### Prerequisites

* Python 3.8+
* Ollama installed locally

---

### Setup

```
git clone <repo-url>
cd project

python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

# Start Ollama model
ollama run gemma:3b

# Run Streamlit app
streamlit run app.py
```

---

## Usage

1. Launch Streamlit app
2. Upload documents
3. Ask questions in chat interface
4. System retrieves relevant context
5. LLM generates grounded response

---

## Limitations

* Naive RAG (no reranking or hybrid search)
* Performance depends on chunk quality
* Limited by local model capability (Gemma 3:4B)
* Agentic chunking is experimental

---

## Future Improvements

* Hybrid search (BM25 + embeddings)
* Reranking models
* Better memory handling
* Multi-document reasoning
* Streaming responses