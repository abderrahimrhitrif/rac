import argparse
import os
import requests
from fetch import GithubFetcher
from preprocess import CodePreprocessor
from db import MilvusManager

class RAC:
    def __init__(self, repo_name: str, tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2", model: str = "Qwen3:0.6B"):
        self.model_name = model
        self.repo_name = repo_name
        parts = repo_name.rstrip('/').split("/")
        self.collection_name = "_".join(parts[-2:])
        self.fetcher = GithubFetcher()
        self.preprocessor = CodePreprocessor()
        self.db_manager = MilvusManager()

    def _load_documents_from_directory(self, directory: str):
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        documents = []
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
        return documents, all_files

    def setup(self):
        if os.path.exists(f"data/{self.collection_name}"):
            print(f"Skipping fetch (data/{self.collection_name} already exists)")
            return
        self.fetcher.fetch(self.repo_name)
        documents, file_paths = self._load_documents_from_directory(f"data/{self.collection_name}")
        print(f"Loaded {len(documents)} documents from {self.collection_name}")
        processed_docs = self.preprocessor.preprocess(documents, file_paths)
        print(f"Processed into {len(processed_docs)} chunks, {len(file_paths)}")
        embeddings = self.preprocessor.embed_documents(processed_docs)
        print(f"Generated embeddings for {len(embeddings)} chunks.")
        if not self.db_manager.client.has_collection(self.collection_name):
            self.db_manager.create_collection(self.collection_name, embeddings[0].shape[1])
        data = self.db_manager.prepare_data(embeddings, processed_docs)
        self.db_manager.insert_data(self.collection_name, data)

    def ask(self, question: str, messages : list = [] ):
        model = self.model_name
        queries = self.prepare_prompt(question, model=model)
        queries = queries if queries else [question]
        retrieved_docs = []
        for query in queries:
            query_embedding = self.preprocessor.get_embedding(query)
            retrieved_doc = self.db_manager.retrieve_documents(self.collection_name, query_embedding.tolist()[0])
            retrieved_docs.extend(retrieved_doc)
        return self._ask_model(user_query=question, retrieved_docs=retrieved_docs, model=model, messages=messages)
    def prepare_prompt(self, user_query: str, model: str):
        """Break down user query into sub-questions using LLM"""
        messages = [
            {"role": "system", "content": "Your task is to break down the user query into smaller sub-questions. If it's already simple, just return it as is. Return them on the following format Question1|Question2|Question3|etc..."},
            {"role": "user", "content": user_query}
        ]

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False}
        )

        if response.status_code == 200:
            content = response.json()["message"]["content"]
            sub_questions = [line.strip()
                             for line in content.split("|") if line.strip()]
            if not sub_questions:
                sub_questions = [user_query]
            return sub_questions
        else:
            print("Error in prepare_prompt:", response.text)
            return [user_query]

    def _ask_model(self, user_query: str, model: str, retrieved_docs: list = None, messages : list = []):
        retrieved_context = "\n".join(
            doc["text"] if isinstance(doc, dict) and "text" in doc else str(doc)
            for doc in (retrieved_docs or [])
        )
        messages = messages + [
            {"role": "system", "content": "You will receive context and must answer using it. If info is missing, say you don't know."},
            {"role": "system", "content": retrieved_context},
            {"role": "system", "content": f"User question: {user_query}"}
        ]

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            }
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            print(response.text)
            return None

