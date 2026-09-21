import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.policy_engine import is_allowed


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_documents():
    with open(DATA_DIR / "documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


class PermissionAwareRetriever:

    def __init__(self, model_name="BAAI/bge-m3"):
        self.documents = load_documents()

        print("Embedding model loading...")
        self.model = SentenceTransformer(model_name)

        self.document_texts = []

        for doc in self.documents:
            text = (
                f"제목: {doc['title']}\n"
                f"내용: {doc['content']}"
            )

            self.document_texts.append(text)

        print("Encoding documents...")

        embeddings = self.model.encode(
            self.document_texts,
            normalize_embeddings=True
        )

        self.document_embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        print(
            f"Document embeddings created: "
            f"{len(self.documents)} documents"
        )

    def get_authorized_indices(self, user_id):
        """
        해당 사용자가 접근할 수 있는 문서의
        인덱스만 반환한다.
        """

        authorized_indices = []

        for index, doc in enumerate(self.documents):

            if is_allowed(
                user_id,
                doc["document_id"]
            ):
                authorized_indices.append(index)

        return authorized_indices

    def search(self, user_id, query, top_k=5):
        """
        1. 사용자의 접근 가능 문서를 먼저 필터링
        2. 허용 문서만 FAISS 인덱스에 포함
        3. 그 안에서 의미 기반 검색 수행
        """

        authorized_indices = self.get_authorized_indices(
            user_id
        )

        if len(authorized_indices) == 0:
            return []

        authorized_embeddings = self.document_embeddings[
            authorized_indices
        ]

        dimension = authorized_embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(authorized_embeddings)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        actual_top_k = min(
            top_k,
            len(authorized_indices)
        )

        scores, local_indices = index.search(
            query_embedding,
            actual_top_k
        )

        results = []

        for score, local_index in zip(
            scores[0],
            local_indices[0]
        ):

            if local_index == -1:
                continue

            global_index = authorized_indices[
                local_index
            ]

            doc = self.documents[global_index]

            results.append(
                {
                    "document_id": doc["document_id"],
                    "document_type": doc["document_type"],
                    "title": doc["title"],
                    "classification": doc["classification"],
                    "score": float(score)
                }
            )

        return results


if __name__ == "__main__":

    retriever = PermissionAwareRetriever()

    query = (
        "ORION-K7 시스템의 "
        "보안점검 결과를 알려줘."
    )

    for user_id in ["U1", "U4"]:

        print(
            f"\n=== {user_id} Retrieval Results ==="
        )

        results = retriever.search(
            user_id=user_id,
            query=query,
            top_k=5
        )

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
