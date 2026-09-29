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

        # 각 테스트가 독립된 Session 상태를 갖도록
        # 매 테스트마다 새 Manager 생성
        self.manager = (
            RevocationAwareSessionManager(
                self.controller
            )
        )

    # =================================================
    # 1. 임시권한 활성 중 공개된 Fact 기록
    # =================================================

    def test_temporary_disclosure_is_recorded(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S1",

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

        self.assertEqual(
            snapshot[
                "revoked_fact_ids"
            ],
            []
        )

    # =================================================
    # 2. 임시권한에 의존한 Rule 기록
    # =================================================

    def test_temporary_rule_dependency_is_recorded(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S2",

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

        snapshot = (
            self.manager
            .get_session_snapshot(
                session_id=
                    "S2",

                user_id=
                    "U3",

                as_of=(
                    "2026-09-29T10:00:00"
                    "+09:00"
                )
            )
        )

        events = snapshot[
            "exposure_events"
        ]

        self.assertEqual(
            len(events),
            2
        )

        for event in events:

            self.assertIn(
                "IR-001",
                event[
                    "temporary_rule_ids"
                ]
            )

            self.assertIn(
                "TA-001",
                event[
                    "active_authorization_ids"
                ]
            )

    # =================================================
    # 3. 임시권한 활성 중에는 재사용 가능
    # =================================================

    def test_fact_reusable_during_authorization(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S3",

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

        revoked = (
            self.manager
            .get_revoked_fact_ids(
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
    # 4. 권한 만료 후 Session reuse 회수
    # =================================================

    def test_fact_revoked_after_expiration(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S4",

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

        revoked = (
            self.manager
            .get_revoked_fact_ids(
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

        self.assertCountEqual(
            revoked,
            [
                "OPS001-F2",
                "OPS002-F2"
            ]
        )

    # =================================================
    # 5. 회수돼도 Exposure History는 삭제하지 않음
    # =================================================

    def test_revocation_does_not_erase_history(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S5",

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

        self.assertCountEqual(
            snapshot[
                "revoked_fact_ids"
            ],
            [
                "OPS001-F2",
                "OPS002-F2"
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
    # 6. 회수 Fact가 Candidate에 다시 등장하면 차단
    # =================================================

    def test_revoked_candidate_is_blocked(
        self
    ):

        self.manager.record_disclosure(
            session_id=
                "S6",

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

        result = (
            self.manager
            .filter_revoked_candidates(
                session_id=
                    "S6",

                user_id=
                    "U3",

                candidate_fact_ids=[
                    "OPS001-F2",
                    "OPS002-F2"
                ],

                as_of=(
                    "2026-09-29T12:30:00"
                    "+09:00"
                )
            )
        )

        self.assertEqual(
            result[
                "reusable_candidate_facts"
            ],
            []
        )

        self.assertCountEqual(
            result[
                "revoked_reuse_facts"
            ],
            [
                "OPS001-F2",
                "OPS002-F2"
            ]
        )

    # =================================================
    # 7. 임시권한과 무관한 공개는 회수하지 않음
    # =================================================

    def test_normal_disclosure_is_not_revoked(
        self
    ):

        # OPS001-F2 하나만 공개하는 것은
        # IR-001을 완성하지 않으므로
        # 임시권한에 의존한 공개가 아니다.

        self.manager.record_disclosure(
            session_id=
                "S7",

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

        revoked = (
            self.manager
            .get_revoked_fact_ids(
                session_id=
                    "S7",

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
            []
        )


if __name__ == "__main__":

    unittest.main()
