import re

from src.policy_engine import is_allowed


SAFE_RESPONSE = (
    "접근 가능한 자료에서 해당 정보를 확인할 수 없습니다."
)


class OutputGuard:

    def __init__(self, documents):
        self.documents = documents

    def get_unauthorized_documents(self, user_id):
        """
        사용자가 접근할 수 없는 문서 목록을 반환한다.
        """

        unauthorized = []

        for doc in self.documents:
            if not is_allowed(
                user_id,
                doc["document_id"]
            ):
                unauthorized.append(doc)

        return unauthorized

    def check_protected_fact_leakage(
        self,
        user_id,
        response
    ):
        """
        비인가 문서에 정의된 protected_fact가
        응답에 직접 나타나는지 검사한다.
        """

        unauthorized_docs = (
            self.get_unauthorized_documents(user_id)
        )

        detected = []

        response_lower = response.lower()

        for doc in unauthorized_docs:

            for fact in doc.get(
                "protected_facts",
                []
            ):

                fact_text = fact["fact"].lower()

                # 전체 사실문 직접 포함 검사
                if fact_text in response_lower:

                    detected.append(
                        {
                            "document_id":
                                doc["document_id"],

                            "fact_id":
                                fact["fact_id"],

                            "category":
                                fact["category"]
                        }
                    )

        return detected

    def extract_identifiers(self, text):
        """
        합성 실험에서 사용하는 코드형 식별자를 추출한다.

        예:
        ZXQ-48291
        SIGMA-17
        OPS-B214
        """

        pattern = (
            r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b"
        )

        return set(
            re.findall(
                pattern,
                text.upper()
            )
        )

    def check_unsupported_identifiers(
        self,
        question,
        context,
        response
    ):
        """
        응답에 등장했지만 질문이나 Context에는 없던
        새로운 코드형 식별자를 탐지한다.
        """

        known_text = (
            question
            + "\n"
            + context
        )

        known_ids = self.extract_identifiers(
            known_text
        )

        response_ids = self.extract_identifiers(
            response
        )

        unsupported = (
            response_ids - known_ids
        )

        return sorted(unsupported)

    def validate(
        self,
        user_id,
        question,
        context,
        response
    ):

        protected_leaks = (
            self.check_protected_fact_leakage(
                user_id,
                response
            )
        )

        unsupported_ids = (
            self.check_unsupported_identifiers(
                question,
                context,
                response
            )
        )

        blocked = (
            len(protected_leaks) > 0
            or len(unsupported_ids) > 0
        )

        if blocked:

            final_response = SAFE_RESPONSE

        else:

            final_response = response

        return {
            "blocked": blocked,
            "protected_leaks": protected_leaks,
            "unsupported_identifiers":
                unsupported_ids,
            "final_response": final_response
        }
