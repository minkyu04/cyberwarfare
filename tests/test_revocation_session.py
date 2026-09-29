import unittest

from src.incremental_disclosure import (
    IncrementalDisclosureController
)

from src.session_control import (
    RevocationAwareSessionManager
)


class TestRevocationAwareSession(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.controller = (
            IncrementalDisclosureController()
        )

    def setUp(self):

        self.manager = (
            RevocationAwareSessionManager(
                self.controller
            )
        )

    # =================================================
    # 공통 임시권한 공개 함수
    # =================================================

    def record_temp_combination(
        self,
        session_id
    ):

        self.manager.record_disclosure(
            session_id=
                session_id,

            user_id=
                "U3",

            disclosed_fact_ids=[
                "OPS001-F2",
                "OPS002-F2"
            ],

            exposed_facts_before=[],

            candidate_facts=[
                "OPS001-F2",
                "OPS002-F2"
            ],

            as_of=(
                "2026-09-29T10:00:00"
                "+09:00"
            )
        )

    # =================================================
    # 1. 임시 공개 기록
    # =================================================

    def test_temporary_disclosure_is_recorded(
        self
    ):

        self.record_temp_combination(
            "S1"
        )

        snapshot = (
            self.manager
            .get_session_snapshot(
                session_id=
                    "S1",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T10:30:00"
                    "+09:00"
                )
            )
        )

        self.assertCountEqual(
            snapshot[
                "exposed_fact_ids"
            ],
            [
                "OPS001-F2",
                "OPS002-F2"
            ]
        )

    # =================================================
    # 2. 임시 Rule 의존성 기록
    # =================================================

    def test_temporary_rule_dependency_is_recorded(
        self
    ):

        self.record_temp_combination(
            "S2"
        )

        snapshot = (
            self.manager
            .get_session_snapshot(
                session_id=
                    "S2",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T10:30:00"
                    "+09:00"
                )
            )
        )

        self.assertIn(
            "IR-001",
            snapshot[
                "temporary_rule_ids"
            ]
        )

    # =================================================
    # 3. 활성 중에는 Rule 회수 없음
    # =================================================

    def test_rule_not_revoked_while_active(
        self
    ):

        self.record_temp_combination(
            "S3"
        )

        revoked = (
            self.manager
            .get_revoked_rule_ids(
                session_id=
                    "S3",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T11:00:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            revoked,
            []
        )

    # =================================================
    # 4. 만료 후 Rule 회수
    # =================================================

    def test_rule_revoked_after_expiration(
        self
    ):

        self.record_temp_combination(
            "S4"
        )

        revoked = (
            self.manager
            .get_revoked_rule_ids(
                session_id=
                    "S4",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            revoked,
            [
                "IR-001"
            ]
        )

    # =================================================
    # 5. Rule이 회수돼도 Exposure History 유지
    # =================================================

    def test_revocation_does_not_erase_history(
        self
    ):

        self.record_temp_combination(
            "S5"
        )

        snapshot = (
            self.manager
            .get_session_snapshot(
                session_id=
                    "S5",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            snapshot[
                "revoked_rule_ids"
            ],
            [
                "IR-001"
            ]
        )

        self.assertCountEqual(
            snapshot[
                "exposed_fact_ids"
            ],
            [
                "OPS001-F2",
                "OPS002-F2"
            ]
        )

    # =================================================
    # 6. 만료 후 개별 안전 Fact 재요청은 허용
    # =================================================

    def test_single_safe_fact_allowed_after_expiration(
        self
    ):

        history = [
            "OPS001-F2",
            "OPS002-F2"
        ]

        result = (
            self.controller
            .evaluate_turn(
                user_id=
                    "U3",

                exposed_facts=
                    history,

                candidate_facts=[
                    "OPS001-F2"
                ],

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            result[
                "current_response_violations"
            ],
            []
        )

        self.assertEqual(
            result[
                "removed_facts"
            ],
            []
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
    # 7. 만료 후 금지조합 전체 재요청 시 선택적 제거
    # =================================================

    def test_combination_replay_is_minimized_after_expiration(
        self
    ):

        history = [
            "OPS001-F2",
            "OPS002-F2"
        ]

        result = (
            self.controller
            .evaluate_turn(
                user_id=
                    "U3",

                exposed_facts=
                    history,

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
                "preexisting_violations"
            ]
        )

        self.assertIn(
            "IR-001",
            result[
                "current_response_violations"
            ]
        )

        # 업무가치:
        #
        # OPS001-F2 = 8
        # OPS002-F2 = 2
        #
        # 둘 중 하나만 제거하면 되므로
        # 업무손실이 작은 OPS002-F2를 제거한다.

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

        self.assertEqual(
            result[
                "removed_utility"
            ],
            2.0
        )

        self.assertEqual(
            result[
                "retained_utility"
            ],
            8.0
        )

    # =================================================
    # 8. 임시권한과 무관한 단독 공개는
    #    Temporary Rule로 기록하지 않음
    # =================================================

    def test_normal_disclosure_has_no_temporary_rule(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S8",

            user_id=
                "U3",

            disclosed_fact_ids=[
                "OPS001-F2"
            ],

            exposed_facts_before=[],

            candidate_facts=[
                "OPS001-F2"
            ],

            as_of=(
                "2026-09-29T10:00:00"
                "+09:00"
            )
        )

        snapshot = (
            self.manager
            .get_session_snapshot(
                session_id=
                    "S8",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            snapshot[
                "temporary_rule_ids"
            ],
            []
        )

        self.assertEqual(
            snapshot[
                "revoked_rule_ids"
            ],
            []
        )


if __name__ == "__main__":

    unittest.main()
    unittest.main()
