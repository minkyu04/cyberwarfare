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
    LLM이 생성한 자연어 응답이 허용된 Fact의 범위를
    벗어나는지 검사한다.

    주요 기능
    1. 허용되지 않은 수치/식별자 생성 탐지
    2. 제거된 Fact의 핵심 토큰 재출력 탐지
    3. 허용된 Fact가 실제 답변에 반영됐는지 확인
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
    # 비교용 텍스트 정규화
    # =================================================

    def normalize_text(
        self,
        text
    ):

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =================================================
    # 응답에서 보안상 의미 있는 토큰 추출
    # =================================================

    def extract_evidence_tokens(
        self,
        text
    ):

        tokens = set()

        # ---------------------------------------------
        # 코드 / 식별자
        # ORION-K7
        # NODE-PAPA
        # ZXQ-48291
        # ---------------------------------------------

        identifier_pattern = (
            r"\b[A-Z]{2,}"
            r"(?:-[A-Z0-9]+)+\b"
        )

        identifiers = re.findall(
            identifier_pattern,
            text.upper()
        )

        tokens.update(
            identifiers
        )

        # ---------------------------------------------
        # 퍼센트
        # 92%
        # 100%
        # ---------------------------------------------

        percentages = re.findall(
            r"\b\d+(?:\.\d+)?\s*%",
            text
        )

        tokens.update(
            item.replace(" ", "")
            for item in percentages
        )

        # ---------------------------------------------
        # 횟수
        # 37회
        # ---------------------------------------------

        counts = re.findall(
            r"\b\d+\s*회\b",
            text
        )

        tokens.update(
            item.replace(" ", "")
            for item in counts
        )

        # ---------------------------------------------
        # 날짜
        # 10월 22일
        # ---------------------------------------------

        dates = re.findall(
            r"\b\d{1,2}\s*월\s*"
            r"\d{1,2}\s*일\b",
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
        # 시간
        # 02시 14분
        # ---------------------------------------------

        times = re.findall(
            r"\b\d{1,2}\s*시"
            r"(?:\s*\d{1,2}\s*분)?\b",
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
        # 긴 숫자열
        # LLM이 임의로 생성한 가짜 식별코드 탐지
        # 예: 0000000000000000
        # ---------------------------------------------

        long_numbers = re.findall(
            r"\b\d{4,}\b",
            text
        )

        tokens.update(
            long_numbers
        )

        # ---------------------------------------------
        # 16진수 코드
        # 예: 0x12345678
        # ---------------------------------------------

        hex_codes = re.findall(
            r"\b0x[0-9a-fA-F]+\b",
            text
        )

        tokens.update(
            code.lower()
            for code in hex_codes
        )

        # ---------------------------------------------
        # 위험등급 표현
        # ---------------------------------------------

        risk_levels = re.findall(
            r"\b(?:Critical|High|Medium|Low)\b",
            text,
            flags=re.IGNORECASE
        )

        tokens.update(
            item.lower()
            for item in risk_levels
        )

        return tokens

    # =================================================
    # 허용된 Source 구성
    # =================================================

    def build_allowed_source(
        self,
        question,
        allowed_fact_ids
    ):

        texts = [
            question
        ]

        for fact_id in allowed_fact_ids:

            texts.append(
                self.get_fact_text(
                    fact_id
                )
            )

        return "\n".join(
            texts
        )

    # =================================================
    # 제거된 Fact 텍스트 구성
    # =================================================

    def build_removed_source(
        self,
        removed_fact_ids
    ):

        texts = []

        for fact_id in removed_fact_ids:

            texts.append(
                self.get_fact_text(
                    fact_id
                )
            )

        return "\n".join(
            texts
        )

    # =================================================
    # 허용 Fact 반영 여부 확인
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

            # 날짜, 시간, 비율, 코드 등
            # 식별 가능한 근거가 있는 Fact에 대해서만
            # 응답 반영 여부를 엄격하게 검사한다.
            if not fact_tokens:
                continue

            if not (
                fact_tokens
                & response_tokens
            ):

                missing.append(
                    fact_id
                )

        return missing

    # =================================================
    # 결정론적 안전 응답 생성
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
        # 허용된 정보의 Evidence Token
        # ---------------------------------------------

        allowed_source = (
            self.build_allowed_source(
                question,
                allowed_fact_ids
            )
        )

        allowed_tokens = (
            self.extract_evidence_tokens(
                allowed_source
            )
        )

        # ---------------------------------------------
        # 실제 응답의 Evidence Token
        # ---------------------------------------------

        response_tokens = (
            self.extract_evidence_tokens(
                draft_response
            )
        )

        # ---------------------------------------------
        # 허용된 Source에 없던 신규 토큰
        # ---------------------------------------------

        unsupported_tokens = sorted(
            response_tokens
            - allowed_tokens
        )

        # ---------------------------------------------
        # 제거된 Fact의 Evidence Token 검사
        # ---------------------------------------------

        removed_source = (
            self.build_removed_source(
                removed_fact_ids
            )
        )

        removed_tokens = (
            self.extract_evidence_tokens(
                removed_source
            )
        )

        question_tokens = (
            self.extract_evidence_tokens(
                question
            )
        )

        leaked_removed_tokens = sorted(
            (
                removed_tokens
                & response_tokens
            )
            - question_tokens
        )

        # ---------------------------------------------
        # 허용 Fact가 실제 응답에 반영됐는지 검사
        # ---------------------------------------------

        missing_allowed_facts = (
            self.find_missing_allowed_facts(
                draft_response,
                allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # 차단 조건
        # ---------------------------------------------

        guard_triggered = (
            bool(unsupported_tokens)
            or bool(
                leaked_removed_tokens
            )
            or bool(
                missing_allowed_facts
            )
        )

        # ---------------------------------------------
        # 검증 실패 시 결정론적 Fallback 사용
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
