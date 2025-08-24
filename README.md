# RAC: Retrieval-Augmented Code

**This project is in early development and not yet fully featured.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAC is a tool that allows you to chat with your codebase. By using Retrieval-Augmented Generation (RAG), RAC can answer your questions about a GitHub repository, providing you with accurate and context-aware answers. This is mainly intended for models with smaller models that lack the context window and paramter count to be aware of a whole repo.

## How it Works

RAC uses a Retrieval-Augmented Generation (RAG) pipeline to answer your questions. Here's a high-level overview of the process:

1.  **Fetch:** RAC fetches the target GitHub repository and extracts its content.
2.  **Preprocess:** The code is split into smaller chunks, and each chunk is embedded using a sentence transformer model.
3.  **Store:** The embeddings are stored in a Milvus database for efficient retrieval.
4.  **Retrieve:** When you ask a question, RAC embeds the question and retrieves the most relevant code snippets from the Milvus database.
5.  **Generate:** The retrieved code snippets are then used as context for a large language model (LLM) to generate an answer.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/abderrahimrhitrif/rac.git
    cd RAC
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use RAC, you need to provide the URL of the GitHub repository and the question you want to ask.

```bash
python main.py <repo_url> <model_name> "<your_question>"
```

For example:

```bash
python main.py fastapi/fastapi Qwen3:0.6B "How does dependency injection work in fast api?"
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
