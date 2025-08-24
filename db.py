from pymilvus import MilvusClient
from tqdm import tqdm

class MilvusManager:
    def __init__(self, uri: str = "milvus_demo.db"):
        self.client = MilvusClient(uri)

    def create_collection(self, collection_name: str, dimension: int):
        if self.client.has_collection(collection_name=collection_name):
            return
        self.client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
        )

    def insert_data(self, collection_name: str, data: list, batch_size: int = 100):
        for i in tqdm(range(0, len(data), batch_size), desc=f"Inserting into {collection_name}"):
            batch = data[i:i+batch_size]
            self.client.insert(collection_name=collection_name, data=batch)

    def retrieve_documents(self, collection_name: str, query_vector: list, limit: int = 10):
        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text"],
        )
        results = []
        hits = res[0]
        for hit in hits:
            results.append(hit['entity']['text'])
        return results

    @staticmethod
    def prepare_data(embeddings, documents):
        data = [
            {"id": i, "vector": embeddings[i].tolist()[0], "text": documents[i], "subject": "code"}
            for i in range(len(embeddings))
        ]
        return data