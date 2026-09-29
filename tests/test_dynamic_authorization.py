import unittest

from src.dynamic_authorization import (
    DynamicAuthorizationManager
)

from src.incremental_disclosure import (
    IncrementalDisclosureController
)


class TestDynamicAuthorization(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.manager = (
            DynamicAuthorizationManager()
        )

        cls.controller = (
            IncrementalDisclosureController()
        )

    # =================================================
    # 1. 임시 권한 시작 전
    # =================================================

    def test_before_authorization_window(
        self
    ):

        context = (
            self.manager
            .get_effective_context(
                user_id="U3",
                as_of=(
                    "2026-09-29T08:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            context[
                "effective_clearance"
            ],
            2
        )

        self.assertEqual(
            context[
                "active_authorization_ids"
            ],
            []
        )

    # =================================================
    # 2. 임시 권한 활성 중
    # =================================================

    def test_active_authorization_window(
        self
    ):

        context = (
            self.manager
            .get_effective_context(
                user_id="U3",
                as_of=(
                    "2026-09-29T10:00:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            context[
                "effective_clearance"
            ],
            3
        )

        self.assertIn(
            "BLUE",
            context[
                "effective_missions"
            ]
        )

        self.assertIn(
            "TA-001",
            context[
                "active_authorization_ids"
            ]
        )

    # =================================================
    # 3. 임시 권한 만료 후
    # =================================================

    def test_after_authorization_window(
        self
    ):

        context = (
            self.manager
            .get_effective_context(
                user_id="U3",
                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            context[
                "effective_clearance"
            ],
            2
        )

        self.assertEqual(
            context[
                "active_authorization_ids"
            ],
            []
        )

    # =================================================
    # 4. valid_until 시각부터 즉시 만료
    # =================================================

    def test_exact_expiration_time(
        self
    ):

        context = (
            self.manager
            .get_effective_context(
                user_id="U3",
                as_of=(
                    "2026-09-29T12:00:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            context[
                "effective_clearance"
            ],
            2
        )

        self.assertEqual(
            context[
                "active_authorization_ids"
            ],
            []
        )

    # =================================================
    # 5. 권한 부여 전에는 정보조합 차단
    # =================================================

    def test_combination_blocked_before_grant(
        self
    ):

        result = (
            self.controller.evaluate_turn(
                user_id="U3",

                exposed_facts=[],

                candidate_facts=[
                    "OPS001-F2",
                    "OPS002-F2"
                ],

                as_of=(
                    "2026-09-29T08:30:00"
                    "+09:00"
                )
            )
        )

        self.assertIn(
            "IR-001",
            result[
                "detected_rules"
            ]
        )

        # OPS002-F2의 업무가치가 더 낮으므로
        # 해당 Fact를 제거해야 한다.
        self.assertEqual(
            result[
                "removed_facts"
            ],
            [
                "OPS002-F2"
            ]
        )

        self.assertEqual(
            result[
                "allowed_facts"
            ],
            [
                "OPS001-F2"
            ]
        )

    # =================================================
    # 6. 권한 활성 중에는 동일 조합 허용
    # =================================================

    def test_combination_allowed_during_grant(
        self
    ):

        result = (
            self.controller.evaluate_turn(
                user_id="U3",

                exposed_facts=[],

                candidate_facts=[
                    "OPS001-F2",
                    "OPS002-F2"
                ],

                as_of=(
                    "2026-09-29T10:00:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            result[
                "detected_rules"
            ],
            []
        )

        self.assertEqual(
            result[
                "removed_facts"
            ],
            []
        )

        self.assertCountEqual(
            result[
                "allowed_facts"
            ],
            [
                "OPS001-F2",
                "OPS002-F2"
            ]
        )

        self.assertIn(
            "TA-001",
            result[
                "authorization_context"
            ][
                "active_authorization_ids"
            ]
        )

    # =================================================
    # 7. 권한 만료 후 다시 차단
    # =================================================

    def test_combination_blocked_after_expiration(
        self
    ):

        result = (
            self.controller.evaluate_turn(
                user_id="U3",

                exposed_facts=[],

                candidate_facts=[
                    "OPS001-F2",
                    "OPS002-F2"
                ],

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertIn(
            "IR-001",
            result[
                "detected_rules"
            ]
        )

        self.assertEqual(
            result[
                "removed_facts"
            ],
            [
                "OPS002-F2"
            ]
        )

    # =================================================
    # 8. 임시 Mission 확장
    # =================================================

    def test_temporary_mission_extension(
        self
    ):

        result = (
            self.controller.evaluate_turn(
                user_id="U4",

                exposed_facts=[],

                candidate_facts=[
                    "VUL001-F3",
                    "VUL002-F3"
                ],

                as_of=(
                    "2026-09-29T15:00:00"
                    "+09:00"
                )
            )
        )

        # 평소 U4는 CYBER이고
        # IR-005는 CYBER_COMMAND가 필요하다.
        #
        # TA-002 활성 중에는
        # CYBER_COMMAND가 추가된다.

        self.assertEqual(
            result[
                "detected_rules"
            ],
            []
        )

        self.assertEqual(
            result[
                "removed_facts"
            ],
            []
        )

        self.assertCountEqual(
            result[
                "allowed_facts"
            ],
            [
                "VUL001-F3",
                "VUL002-F3"
            ]
        )

        self.assertIn(
            "TA-002",
            result[
                "authorization_context"
            ][
                "active_authorization_ids"
            ]
        )

    # =================================================
    # 9. 기존 실험 호환성
    # =================================================

    def test_no_timestamp_preserves_baseline(
        self
    ):

        result = (
            self.controller.evaluate_turn(
                user_id="U3",

                exposed_facts=[],

                candidate_facts=[
                    "OPS001-F2",
                    "OPS002-F2"
                ]
            )
        )

        # as_of가 없으면 기존 연구조건과 동일하게
        # 임시 권한을 적용하지 않는다.

        self.assertIn(
            "IR-001",
            result[
                "detected_rules"
            ]
        )

        self.assertEqual(
            result[
                "authorization_context"
            ][
                "active_authorization_ids"
            ],
            []
        )


if __name__ == "__main__":

    unittest.main()
