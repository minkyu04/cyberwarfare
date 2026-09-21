import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_documents():
    with open(DATA_DIR / "documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


class DocumentRetriever:

    def __init__(self, model_name="BAAI/bge-m3"):
        self.documents = load_documents()

        print("Embedding model loading...")
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.document_embeddings = None

        self.build_index()

    def build_index(self):
        """
        문서 제목과 본문을 임베딩하여 FAISS 인덱스를 생성한다.
        """

        texts = []

        for doc in self.documents:
            text = (
                f"제목: {doc['title']}\n"
                f"내용: {doc['content']}"
            )

            texts.append(text)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.document_embeddings = embeddings

        dimension = embeddings.shape[1]

        # cosine similarity와 동일한 방식으로 검색
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(
            f"FAISS index created: "
            f"{self.index.ntotal} documents"
        )

    def search(self, query, top_k=5):
        """
        질문과 가장 유사한 문서를 검색한다.
        """

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            doc = self.documents[index]

            results.append(
                {
                    "document_id": doc["document_id"],
                    "document_type": doc["document_type"],
                    "title": doc["title"],
                    "score": float(score)
                }
            )

        return results


if __name__ == "__main__":

    retriever = DocumentRetriever()

    query = "ORION-K7 시스템의 보안점검 결과를 알려줘."

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n=== Retrieval Results ===")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['document_id']} | "
            f"{result['title']} | "
            f"score={result['score']:.4f}"
        )
