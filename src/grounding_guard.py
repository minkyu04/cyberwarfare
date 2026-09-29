import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


SAFE_RESPONSE = (
    "현재 접근 가능한 정보와 정보결합 정책을 기준으로 "
    "해당 내용을 제공할 수 없습니다."
)


def load_json(filename):

    with open(
        DATA_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


class GroundedOutputGuard:
    """
    LLM 응답이 현재 허용된 근거를 벗어나는지 검사한다.

    검증 대상
    --------------------------------------------------
    1. 허용되지 않은 수치/식별자 생성
    2. 제거된 Fact 재출력
    3. 허용된 Fact 누락
    4. 공개문서 질의에서 Context에 없는 값의 환각

    보호 Fact 질의뿐 아니라 공개문서 응답에도
    동일한 Grounding 검사를 적용할 수 있다.
    """

    def __init__(self):

        self.documents = load_json(
            "documents.json"
        )

        self.fact_map = {}

        for doc in self.documents:

            for fact in doc.get(
                "protected_facts",
                []
            ):

                self.fact_map[
                    fact["fact_id"]
                ] = {
                    "fact_id":
                        fact["fact_id"],

                    "fact":
                        fact["fact"],

                    "category":
                        fact["category"],

                    "weight":
                        fact["weight"],

                    "document_id":
                        doc["document_id"],

                    "title":
                        doc["title"]
                }

    # =================================================
    # Fact
    # =================================================

    def get_fact_text(
        self,
        fact_id
    ):

        if fact_id not in self.fact_map:

            raise ValueError(
                f"Unknown fact: {fact_id}"
            )

        return self.fact_map[
            fact_id
        ]["fact"]

    # =================================================
    # Evidence Token
    # =================================================

    def extract_evidence_tokens(
        self,
        text
    ):

        tokens = set()

        if not text:

            return tokens

        upper_text = text.upper()

        # ---------------------------------------------
        # 영문 식별자
        #
        # NODE-PAPA
        # ORION-K7
        # ZXQ-48291
        # ---------------------------------------------

        identifier_pattern = (
            r"(?<![A-Z0-9])"
            r"([A-Z]{2,}"
            r"(?:-[A-Z0-9]+)+)"
            r"(?![A-Z0-9-])"
        )

        identifiers = re.findall(
            identifier_pattern,
            upper_text
        )

        tokens.update(
            identifiers
        )

        # ---------------------------------------------
        # 퍼센트
        # ---------------------------------------------

        percentages = re.findall(
            r"(?<![\d.])"
            r"\d+(?:\.\d+)?\s*%",
            text
        )

        tokens.update(
            item.replace(
                " ",
                ""
            )
            for item in percentages
        )

        # ---------------------------------------------
        # 횟수
        # ---------------------------------------------

        counts = re.findall(
            r"(?<!\d)"
            r"\d+\s*회",
            text
        )

        tokens.update(
            item.replace(
                " ",
                ""
            )
            for item in counts
        )

        # ---------------------------------------------
        # 날짜
        # ---------------------------------------------

        dates = re.findall(
            r"(?<!\d)"
            r"\d{1,2}\s*월\s*"
            r"\d{1,2}\s*일",
            text
        )

        tokens.update(
            re.sub(
                r"\s+",
                "",
                item
            )
            for item in dates
        )

        # ---------------------------------------------
        # 시각
        # ---------------------------------------------

        times = re.findall(
            r"(?<!\d)"
            r"\d{1,2}\s*시"
            r"(?:\s*\d{1,2}\s*분)?",
            text
        )

        tokens.update(
            re.sub(
                r"\s+",
                "",
                item
            )
            for item in times
        )

        # ---------------------------------------------
        # 연도
        #
        # 2023년 등
        # ---------------------------------------------

        years = re.findall(
            r"(?<!\d)"
            r"\d{4}\s*년",
            text
        )

        tokens.update(
            item.replace(
                " ",
                ""
            )
            for item in years
        )

        # ---------------------------------------------
        # 긴 숫자열
        # ---------------------------------------------

        long_numbers = re.findall(
            r"(?<![A-Za-z0-9-])"
            r"\d{4,}"
            r"(?![A-Za-z0-9-])",
            text
        )

        tokens.update(
            long_numbers
        )

        # ---------------------------------------------
        # 16진수
        # ---------------------------------------------

        hex_codes = re.findall(
            r"(?<![0-9a-fA-F])"
            r"0x[0-9a-fA-F]+"
            r"(?![0-9a-fA-F])",
            text
        )

        tokens.update(
            code.lower()
            for code in hex_codes
        )

        # ---------------------------------------------
        # 위험도
        # ---------------------------------------------

        risk_levels = re.findall(
            r"(?<![A-Za-z])"
            r"(?:Critical|High|Medium|Low)"
            r"(?![A-Za-z])",
            text,
            flags=re.IGNORECASE
        )

        tokens.update(
            item.lower()
            for item in risk_levels
        )

        return tokens

    # =================================================
    # 허용 Fact Token
    # =================================================

    def get_allowed_fact_tokens(
        self,
        allowed_fact_ids
    ):

        tokens = set()

        for fact_id in (
            allowed_fact_ids
        ):

            tokens.update(
                self.extract_evidence_tokens(
                    self.get_fact_text(
                        fact_id
                    )
                )
            )

        return tokens

    # =================================================
    # 제거 Fact Token
    # =================================================

    def get_removed_fact_tokens(
        self,
        removed_fact_ids
    ):

        tokens = set()

        for fact_id in (
            removed_fact_ids
        ):

            tokens.update(
                self.extract_evidence_tokens(
                    self.get_fact_text(
                        fact_id
                    )
                )
            )

        return tokens

    # =================================================
    # 허용 Fact 누락
    # =================================================

    def find_missing_allowed_facts(
        self,
        response,
        allowed_fact_ids
    ):

        response_tokens = (
            self.extract_evidence_tokens(
                response
            )
        )

        missing = []

        for fact_id in (
            allowed_fact_ids
        ):

            fact_tokens = (
                self.extract_evidence_tokens(
                    self.get_fact_text(
                        fact_id
                    )
                )
            )

            # 검증할 수 있는 token이 없는 Fact는
            # 현재 Guard의 엄격한 누락검사에서 제외
            if not fact_tokens:

                continue

            if not fact_tokens.issubset(
                response_tokens
            ):

                missing.append(
                    fact_id
                )

        return missing

    # =================================================
    # Fact 기반 Fallback
    # =================================================

    def build_grounded_fallback(
        self,
        allowed_fact_ids
    ):

        if not allowed_fact_ids:

            return SAFE_RESPONSE

        facts = [
            self.get_fact_text(
                fact_id
            )
            for fact_id
            in allowed_fact_ids
        ]

        return (
            "확인 가능한 정보는 다음과 같습니다. "
            + " ".join(
                facts
            )
        )

    # =================================================
    # 검증
    # =================================================

    def validate(
        self,
        question,
        allowed_fact_ids,
        removed_fact_ids,
        draft_response,
        additional_allowed_text="",
        fallback_response=None
    ):

        # ---------------------------------------------
        # 사용자 질문에 이미 존재했던 정보
        # ---------------------------------------------

        question_tokens = (
            self.extract_evidence_tokens(
                question
            )
        )

        # ---------------------------------------------
        # 정책상 허용 Fact
        # ---------------------------------------------

        allowed_fact_tokens = (
            self.get_allowed_fact_tokens(
                allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # 실제 Context에 존재하는 추가 근거
        #
        # 공개문서 Context 등을 지원
        # ---------------------------------------------

        context_tokens = (
            self.extract_evidence_tokens(
                additional_allowed_text
            )
        )

        # ---------------------------------------------
        # 제거된 Fact
        # ---------------------------------------------

        removed_fact_tokens = (
            self.get_removed_fact_tokens(
                removed_fact_ids
            )
        )

        # ---------------------------------------------
        # 생성응답
        # ---------------------------------------------

        response_tokens = (
            self.extract_evidence_tokens(
                draft_response
            )
        )

        # ---------------------------------------------
        # Grounded Token 집합
        # ---------------------------------------------

        supported_tokens = (
            question_tokens
            | allowed_fact_tokens
            | context_tokens
        )

        # ---------------------------------------------
        # 근거 없는 신규 token
        # ---------------------------------------------

        unsupported_tokens = sorted(
            response_tokens
            - supported_tokens
        )

        # ---------------------------------------------
        # 제거된 정보 재출력
        #
        # 사용자 질문에 이미 있거나,
        # 현재 허용된 근거에도 존재하는 token은
        # 신규 유출로 계산하지 않는다.
        # ---------------------------------------------

        leaked_removed_tokens = sorted(
            (
                removed_fact_tokens
                & response_tokens
            )
            - question_tokens
            - allowed_fact_tokens
            - context_tokens
        )

        # ---------------------------------------------
        # 허용 Fact 누락
        # ---------------------------------------------

        missing_allowed_facts = (
            self.find_missing_allowed_facts(
                response=
                    draft_response,

                allowed_fact_ids=
                    allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # Guard 여부
        # ---------------------------------------------

        guard_triggered = (
            bool(
                unsupported_tokens
            )
            or bool(
                leaked_removed_tokens
            )
            or bool(
                missing_allowed_facts
            )
        )

        # ---------------------------------------------
        # 최종 응답
        # ---------------------------------------------

        if guard_triggered:

            if fallback_response is not None:

                final_response = (
                    fallback_response
                )

            else:

                final_response = (
                    self.build_grounded_fallback(
                        allowed_fact_ids
                    )
                )

            response_mode = (
                "GROUNDED_FALLBACK"
            )

        else:

            final_response = (
                draft_response.strip()
            )

            response_mode = (
                "LLM_PASSED"
            )

        return {
            "guard_triggered":
                guard_triggered,

            "unsupported_tokens":
                unsupported_tokens,

            "leaked_removed_tokens":
                leaked_removed_tokens,

            "missing_allowed_facts":
                missing_allowed_facts,

            "response_mode":
                response_mode,

            "draft_response":
                draft_response.strip(),

            "final_response":
                final_response
        }
