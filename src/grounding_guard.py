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
    LLM 자연어 응답이 허용된 Fact 범위를 벗어나는지 검사한다.

    기능
    1. 허용되지 않은 수치 및 식별자 생성 탐지
    2. 제거된 Fact의 핵심 정보 재출력 탐지
    3. 허용된 Fact가 실제 답변에 반영됐는지 검사
    4. 검증 실패 시 Fact 기반 결정론적 응답으로 대체
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
    # Fact 조회
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
    # Evidence Token 추출
    # =================================================

    def extract_evidence_tokens(
        self,
        text
    ):
        """
        자연어에서 보안상 검증 가능한 핵심 값을 추출한다.

        Python의 \\b는 Unicode 기준으로 동작하기 때문에
        'NODE-PAPA입니다', '37회입니다'와 같이
        한국어 조사가 바로 붙으면 경계 인식에 문제가 생긴다.

        따라서 ASCII 기반 lookaround를 사용한다.
        """

        tokens = set()

        upper_text = text.upper()

        # ---------------------------------------------
        # 1. 영문 식별자
        #
        # NODE-PAPA
        # NOVA-17
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
        # 2. 퍼센트
        #
        # 92%
        # 100%
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
        # 3. 횟수
        #
        # 37회
        # 37회입니다
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
        # 4. 날짜
        #
        # 10월 22일
        # 10월 22일입니다
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
        # 5. 시각
        #
        # 02시
        # 02시 14분
        # 02시 14분입니다
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
        # 6. 긴 숫자열
        #
        # 00000000
        # 12345678
        #
        # 단, ZXQ-48291 같은 식별자 내부 숫자는 제외
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
        # 7. 16진수 코드
        #
        # 0x12345678
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
        # 8. 위험등급
        #
        # Critical
        # High
        # Medium
        # Low
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
    # 허용된 Fact Token
    # =================================================

    def get_allowed_fact_tokens(
        self,
        allowed_fact_ids
    ):

        tokens = set()

        for fact_id in allowed_fact_ids:

            fact_text = (
                self.get_fact_text(
                    fact_id
                )
            )

            tokens.update(
                self.extract_evidence_tokens(
                    fact_text
                )
            )

        return tokens

    # =================================================
    # 제거된 Fact Token
    # =================================================

    def get_removed_fact_tokens(
        self,
        removed_fact_ids
    ):

        tokens = set()

        for fact_id in removed_fact_ids:

            fact_text = (
                self.get_fact_text(
                    fact_id
                )
            )

            tokens.update(
                self.extract_evidence_tokens(
                    fact_text
                )
            )

        return tokens

    # =================================================
    # 허용된 Fact 반영 여부 검사
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

        for fact_id in allowed_fact_ids:

            fact_text = (
                self.get_fact_text(
                    fact_id
                )
            )

            fact_tokens = (
                self.extract_evidence_tokens(
                    fact_text
                )
            )

            # 현재 Guard는 날짜, 시간, 수치,
            # 식별자 등 검증 가능한 Evidence Token이
            # 있는 Fact를 대상으로 한다.
            if not fact_tokens:
                continue

            # 기존 방식:
            # 하나라도 겹치면 Fact가 출력된 것으로 판단
            #
            # 수정 방식:
            # Fact의 검증 가능한 핵심 토큰이 모두
            # 응답에 존재해야 한다.
            if not fact_tokens.issubset(
                response_tokens
            ):

                missing.append(
                    fact_id
                )

        return missing

    # =================================================
    # 결정론적 Grounded Fallback
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
            + " ".join(facts)
        )

    # =================================================
    # 최종 응답 검증
    # =================================================

    def validate(
        self,
        question,
        allowed_fact_ids,
        removed_fact_ids,
        draft_response
    ):

        # ---------------------------------------------
        # 질문에 이미 포함된 정보
        #
        # 사용자가 질문에서 직접 제시한 값은
        # 시스템의 신규 유출로 보지 않는다.
        # ---------------------------------------------

        question_tokens = (
            self.extract_evidence_tokens(
                question
            )
        )

        # ---------------------------------------------
        # 허용된 Fact
        # ---------------------------------------------

        allowed_fact_tokens = (
            self.get_allowed_fact_tokens(
                allowed_fact_ids
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
        # 실제 LLM 응답
        # ---------------------------------------------

        response_tokens = (
            self.extract_evidence_tokens(
                draft_response
            )
        )

        # ---------------------------------------------
        # 허용되지 않은 신규 Evidence
        # ---------------------------------------------

        supported_tokens = (
            question_tokens
            | allowed_fact_tokens
        )

        unsupported_tokens = sorted(
            response_tokens
            - supported_tokens
        )

        # ---------------------------------------------
        # 제거된 Fact 재생성 탐지
        #
        # 단,
        # 1. 사용자가 질문에서 이미 제공한 값
        # 2. 동시에 허용된 다른 Fact에도 포함된 값
        #
        # 은 신규 누출로 판단하지 않는다.
        # ---------------------------------------------

        leaked_removed_tokens = sorted(
            (
                removed_fact_tokens
                & response_tokens
            )
            - question_tokens
            - allowed_fact_tokens
        )

        # ---------------------------------------------
        # 허용된 Fact 누락 여부
        # ---------------------------------------------

        missing_allowed_facts = (
            self.find_missing_allowed_facts(
                draft_response,
                allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # Guard 작동 조건
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
        # 최종 응답 결정
        # ---------------------------------------------

        if guard_triggered:

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
