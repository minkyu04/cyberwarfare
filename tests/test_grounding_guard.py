import unittest

from src.grounding_guard import (
    GroundedOutputGuard,
    SAFE_RESPONSE
)


class TestGroundedOutputGuard(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.guard = (
            GroundedOutputGuard()
        )

    # =================================================
    # 1. 정상 Fact
    # =================================================

    def test_allowed_fact_response_passes(
        self
    ):

        result = self.guard.validate(
            question=(
                "BLUE 임무 훈련일을 알려줘."
            ),

            allowed_fact_ids=[
                "OPS001-F2"
            ],

            removed_fact_ids=[],

            draft_response=(
                "BLUE 임무 훈련일은 "
                "10월 22일입니다."
            )
        )

        self.assertFalse(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "LLM_PASSED"
        )

    # =================================================
    # 2. 가짜 비율
    # =================================================

    def test_hallucinated_percentage_is_blocked(
        self
    ):

        result = self.guard.validate(
            question=(
                "BLUE 훈련일과 "
                "통신장비 점검률을 알려줘."
            ),

            allowed_fact_ids=[
                "OPS001-F2"
            ],

            removed_fact_ids=[
                "OPS002-F2"
            ],

            draft_response=(
                "훈련일은 10월 22일이며 "
                "통신장비 점검률은 "
                "100%입니다."
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertIn(
            "100%",
            result[
                "unsupported_tokens"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "GROUNDED_FALLBACK"
        )

    # =================================================
    # 3. 제거된 Fact 재생성
    # =================================================

    def test_removed_fact_leakage_is_blocked(
        self
    ):

        result = self.guard.validate(
            question=(
                "BLUE 훈련일과 점검률, "
                "추가 확인 대상을 알려줘."
            ),

            allowed_fact_ids=[
                "OPS001-F2"
            ],

            removed_fact_ids=[
                "OPS002-F2",
                "OPS002-F3"
            ],

            draft_response=(
                "훈련일은 10월 22일이며 "
                "점검률은 92%이고 "
                "대상은 NODE-PAPA입니다."
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertIn(
            "92%",
            result[
                "leaked_removed_tokens"
            ]
        )

        self.assertIn(
            "NODE-PAPA",
            result[
                "leaked_removed_tokens"
            ]
        )

        self.assertNotIn(
            "92%",
            result[
                "final_response"
            ]
        )

        self.assertNotIn(
            "NODE-PAPA",
            result[
                "final_response"
            ]
        )

    # =================================================
    # 4. 허용 Fact 누락
    # =================================================

    def test_missing_allowed_facts_are_restored(
        self
    ):

        result = self.guard.validate(
            question=(
                "보안 이벤트 탐지정보와 "
                "인증 실패 현황을 알려줘."
            ),

            allowed_fact_ids=[
                "LOG001-F3",
                "LOG002-F3"
            ],

            removed_fact_ids=[],

            draft_response=(
                "현재 정보만으로는 "
                "답변하기 어렵습니다."
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertCountEqual(
            result[
                "missing_allowed_facts"
            ],
            [
                "LOG001-F3",
                "LOG002-F3"
            ]
        )

        self.assertIn(
            "02시 14분",
            result[
                "final_response"
            ]
        )

        self.assertIn(
            "37회",
            result[
                "final_response"
            ]
        )

    # =================================================
    # 5. 복수 Fact 정상
    # =================================================

    def test_multiple_allowed_facts_pass(
        self
    ):

        result = self.guard.validate(
            question=(
                "탐지 시각과 "
                "인증 실패 횟수를 알려줘."
            ),

            allowed_fact_ids=[
                "LOG001-F3",
                "LOG002-F3"
            ],

            removed_fact_ids=[],

            draft_response=(
                "탐지 시각은 "
                "02시 14분이며 "
                "인증 실패 횟수는 "
                "37회입니다."
            )
        )

        self.assertFalse(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "LLM_PASSED"
        )

    # =================================================
    # 6. 허용 Fact 없음
    # =================================================

    def test_no_allowed_fact_uses_safe_response(
        self
    ):

        result = (
            self.guard
            .build_grounded_fallback(
                []
            )
        )

        self.assertEqual(
            result,
            SAFE_RESPONSE
        )

    # =================================================
    # 7. 한국어 조사 + 영문 ID
    # =================================================

    def test_identifier_with_korean_suffix(
        self
    ):

        tokens = (
            self.guard
            .extract_evidence_tokens(
                "대상은 "
                "NODE-PAPA입니다."
            )
        )

        self.assertIn(
            "NODE-PAPA",
            tokens
        )

    # =================================================
    # 8. 한국어 조사 + 수치
    # =================================================

    def test_numeric_tokens_with_korean_suffix(
        self
    ):

        tokens = (
            self.guard
            .extract_evidence_tokens(
                "탐지 시각은 "
                "02시 14분이며 "
                "실패 횟수는 "
                "37회입니다."
            )
        )

        self.assertIn(
            "02시14분",
            tokens
        )

        self.assertIn(
            "37회",
            tokens
        )

    # =================================================
    # 9. 공개 Context의 값은 정상 허용
    # =================================================

    def test_public_context_token_is_allowed(
        self
    ):

        public_context = (
            "정기 체육행사는 "
            "10월 5일에 실시한다."
        )

        result = self.guard.validate(
            question=(
                "정기 체육행사 날짜를 알려줘."
            ),

            allowed_fact_ids=[],

            removed_fact_ids=[],

            draft_response=(
                "정기 체육행사는 "
                "10월 5일에 실시합니다."
            ),

            additional_allowed_text=
                public_context,

            fallback_response=(
                "확인 가능한 공개정보: "
                + public_context
            )
        )

        self.assertFalse(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "LLM_PASSED"
        )

    # =================================================
    # 10. 공개 Context에 없는 환각값 차단
    # =================================================

    def test_public_context_hallucination_is_blocked(
        self
    ):

        public_context = (
            "정기 체육행사는 "
            "10월 5일에 실시한다."
        )

        fallback = (
            "확인 가능한 공개정보: "
            + public_context
        )

        result = self.guard.validate(
            question=(
                "정기 체육행사 날짜를 알려줘."
            ),

            allowed_fact_ids=[],

            removed_fact_ids=[],

            draft_response=(
                "정기 체육행사는 "
                "10월 5일이며 "
                "참가율은 95%입니다."
            ),

            additional_allowed_text=
                public_context,

            fallback_response=
                fallback
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertIn(
            "95%",
            result[
                "unsupported_tokens"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "GROUNDED_FALLBACK"
        )

        self.assertNotIn(
            "95%",
            result[
                "final_response"
            ]
        )


if __name__ == "__main__":

    unittest.main()
    
