import unittest

from src.grounding_guard import (
    GroundedOutputGuard,
    SAFE_RESPONSE
)


class TestGroundedOutputGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.guard = (
            GroundedOutputGuard()
        )

    # =================================================
    # 1. 정상 허용 Fact 응답
    # =================================================

    def test_allowed_fact_response_passes(
        self
    ):

        result = (
            self.guard.validate(
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

        self.assertEqual(
            result[
                "unsupported_tokens"
            ],
            []
        )

        self.assertEqual(
            result[
                "missing_allowed_facts"
            ],
            []
        )

        self.assertEqual(
            result[
                "final_response"
            ],
            (
                "BLUE 임무 훈련일은 "
                "10월 22일입니다."
            )
        )

    # =================================================
    # 2. LLM 가짜 비율 생성
    # =================================================

    def test_hallucinated_percentage_is_blocked(
        self
    ):

        result = (
            self.guard.validate(
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
                    "BLUE 훈련일은 "
                    "10월 22일이며 "
                    "통신장비 점검률은 "
                    "100%입니다."
                )
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "GROUNDED_FALLBACK"
        )

        self.assertIn(
            "100%",
            result[
                "unsupported_tokens"
            ]
        )

        self.assertEqual(
            result[
                "final_response"
            ],
            (
                "확인 가능한 정보는 다음과 같습니다. "
                "훈련일은 10월 22일이다"
            )
        )

    # =================================================
    # 3. 제거된 Fact 재출력
    # =================================================

    def test_removed_fact_leakage_is_blocked(
        self
    ):

        result = (
            self.guard.validate(
                question=(
                    "BLUE 훈련일과 "
                    "통신장비 점검률, "
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
                    "훈련일은 "
                    "10월 22일이며 "
                    "통신장비 점검률은 "
                    "92%이고 "
                    "추가 확인 대상은 "
                    "NODE-PAPA입니다."
                )
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "GROUNDED_FALLBACK"
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

        self.assertIn(
            "10월 22일",
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

        result = (
            self.guard.validate(
                question=(
                    "NOVA-17 보안 이벤트 "
                    "탐지정보와 "
                    "AURORA-GATE 인증 실패 "
                    "현황을 알려줘."
                ),

                allowed_fact_ids=[
                    "LOG001-F3",
                    "LOG002-F3"
                ],

                removed_fact_ids=[],

                draft_response=(
                    "현재 제공된 정보만으로는 "
                    "답변하기 어렵습니다."
                )
            )
        )

        self.assertTrue(
            result[
                "guard_triggered"
            ]
        )

        self.assertEqual(
            result[
                "response_mode"
            ],
            "GROUNDED_FALLBACK"
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
    # 5. 여러 허용 Fact 정상 통과
    # =================================================

    def test_multiple_allowed_facts_pass(
        self
    ):

        result = (
            self.guard.validate(
                question=(
                    "보안 이벤트 탐지 시각과 "
                    "인증 실패 횟수를 알려줘."
                ),

                allowed_fact_ids=[
                    "LOG001-F3",
                    "LOG002-F3"
                ],

                removed_fact_ids=[],

                draft_response=(
                    "보안 이벤트 탐지 시각은 "
                    "02시 14분이며, "
                    "인증 실패 횟수는 "
                    "37회입니다."
                )
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

        self.assertEqual(
            result[
                "unsupported_tokens"
            ],
            []
        )

        self.assertEqual(
            result[
                "missing_allowed_facts"
            ],
            []
        )

    # =================================================
    # 6. 허용 Fact 없음
    # =================================================

    def test_no_allowed_fact_uses_safe_response(
        self
    ):

        fallback = (
            self.guard.build_grounded_fallback(
                []
            )
        )

        self.assertEqual(
            fallback,
            SAFE_RESPONSE
        )

    # =================================================
    # 7. 영문 식별자 뒤 한국어 조사 처리
    # =================================================

    def test_identifier_with_korean_suffix(
        self
    ):

        tokens = (
            self.guard.extract_evidence_tokens(
                "추가 확인 대상은 "
                "NODE-PAPA입니다."
            )
        )

        self.assertIn(
            "NODE-PAPA",
            tokens
        )

    # =================================================
    # 8. 수치 뒤 한국어 조사 처리
    # =================================================

    def test_numeric_tokens_with_korean_suffix(
        self
    ):

        tokens = (
            self.guard.extract_evidence_tokens(
                "탐지 시각은 "
                "02시 14분이며, "
                "인증 실패 횟수는 "
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


if __name__ == "__main__":

    unittest.main()
