import unittest

from src.permission_retrieval import PermissionAwareRetriever
from src.policy_engine import is_allowed


class TestPermissionAwareRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 임베딩 모델은 테스트 전체에서 한 번만 로드
        cls.retriever = PermissionAwareRetriever()

    def test_u1_cannot_retrieve_vulnerability_report(self):
        query = "ORION-K7 시스템의 보안점검 결과를 알려줘."

        results = self.retriever.search(
            user_id="U1",
            query=query,
            top_k=5
        )

        document_ids = [
            result["document_id"]
            for result in results
        ]

        self.assertNotIn(
            "VUL-001",
            document_ids
        )

    def test_u4_can_retrieve_vulnerability_report(self):
        query = "ORION-K7 시스템의 보안점검 결과를 알려줘."

        results = self.retriever.search(
            user_id="U4",
            query=query,
            top_k=5
        )

        document_ids = [
            result["document_id"]
            for result in results
        ]

        self.assertIn(
            "VUL-001",
            document_ids
        )

    def test_all_retrieved_documents_are_authorized(self):

        test_queries = {
            "U1": "체육행사 일정을 알려줘.",
            "U2": "ECHO-21 인사배치 검토 내용을 알려줘.",
            "U3": "BLUE 임무 훈련계획을 알려줘.",
            "U4": "ORION-K7 보안점검 결과를 알려줘."
        }

        for user_id, query in test_queries.items():

            results = self.retriever.search(
                user_id=user_id,
                query=query,
                top_k=5
            )

            for result in results:

                document_id = result["document_id"]

                self.assertTrue(
                    is_allowed(
                        user_id,
                        document_id
                    ),
                    msg=(
                        f"Unauthorized retrieval detected: "
                        f"{user_id} -> {document_id}"
                    )
                )


if __name__ == "__main__":
    unittest.main()
