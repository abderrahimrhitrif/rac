# RAC — Retrieval-Augmented Code

## Overview
RAC (Retrieval-Augmented Code) is a CLI tool that enables developers to chat with any GitHub repository by fetching source code, chunking it, embedding it into a Milvus vector database, and generating context-grounded answers using local LLMs.

## Architecture & Pipeline
1. **Fetch (`fetch.py`)**: Downloads and extracts target GitHub repository source code.
2. **Preprocess (`preprocess.py`)**: Language-aware code splitting (functions, classes, paragraphs), token chunking via tiktoken / sentence-transformers, and embedding generation.
3. **Store (`db.py`)**: Manages Milvus vector database collections for storing code vector embeddings.
4. **Query & Generate (`rac.py`, `main.py`)**: Breaks down user queries into sub-questions, retrieves relevant code chunks from Milvus, and queries Ollama local models with grounded context.

## Data
- **Source**: Target GitHub repositories fetched via API or archive downloads.

## Method & Algorithms
- **Code Retrieval RAG**: Vector similarity search over embedded source code chunks.
- **Query Decomposition**: LLM-based sub-question generation to improve multi-part code retrieval recall.

## Verified Results & Metrics
- Implemented full end-to-end RAG pipeline for source code repositories utilizing Milvus vector storage and Ollama local LLM inference.

## Limitations & Caveats
- Early development CLI tool; depends on local Ollama and Milvus service availability.

## Run Instructions
```bash
pip install -r requirements.txt
python main.py <repo_url> <model_name> "<question>"
```

## CV Bullet Points
- Developed RAC (Retrieval-Augmented Code), a CLI tool to chat with GitHub repositories using a full RAG pipeline (repo fetching, language-aware code chunking, sentence-transformers, Milvus vector database, Ollama).
- Implemented query decomposition to break down complex queries into sub-questions for enhanced code retrieval precision.

## Claims to Avoid
- Do not claim commercial developer IDE plugin deployment.
