import unittest

from src.incremental_disclosure import (
    IncrementalDisclosureController
)


class TestIncrementalDisclosureController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = IncrementalDisclosureController()

    # -------------------------------------------------
    # 1. 다중턴 누적 유출 탐지
    # -------------------------------------------------

    def test_multiturn_violation_detection(self):

        result = self.controller.evaluate_turn(
            user_id="U3",
            exposed_facts=[
                "OPS001-F2"
            ],
            candidate_facts=[
                "OPS002-F2"
            ]
        )

        self.assertIn(
            "IR-001",
            result["detected_rules"]
        )

        self.assertEqual(
            result["removed_facts"],
            ["OPS002-F2"]
        )

        self.assertEqual(
            result["allowed_facts"],
            []
        )

    # -------------------------------------------------
    # 2. 최소 제거 집합 검사
    # -------------------------------------------------

    def test_minimum_removal(self):

        result = self.controller.evaluate_turn(
            user_id="U3",
            exposed_facts=[],
            candidate_facts=[
                "OPS001-F2",
                "OPS002-F2",
                "OPS002-F3"
            ]
        )

        self.assertCountEqual(
            result["detected_rules"],
            [
                "IR-001",
                "IR-002"
            ]
        )

        # OPS001-F2 하나를 제거하면
        # IR-001과 IR-002를 동시에 차단할 수 있음
        self.assertEqual(
            result["removed_facts"],
            ["OPS001-F2"]
        )

        self.assertCountEqual(
            result["allowed_facts"],
            [
                "OPS002-F2",
                "OPS002-F3"
            ]
        )

    # -------------------------------------------------
    # 3. 정상 권한 사용자의 정보 보존
    # -------------------------------------------------

    def test_authorized_combination_is_preserved(self):

        result = self.controller.evaluate_turn(
            user_id="U4",
            exposed_facts=[],
            candidate_facts=[
                "LOG001-F3",
                "LOG002-F3"
            ]
        )

        self.assertEqual(
            result["detected_rules"],
            []
        )

        self.assertEqual(
            result["removed_facts"],
            []
        )

        self.assertCountEqual(
            result["allowed_facts"],
            [
                "LOG001-F3",
                "LOG002-F3"
            ]
        )

    # -------------------------------------------------
    # 4. 같은 Clearance라도 Mission 불일치 시 제한
    # -------------------------------------------------

    def test_mission_mismatch(self):

        result = self.controller.evaluate_turn(
            user_id="U4",
            exposed_facts=[],
            candidate_facts=[
                "VUL001-F3",
                "VUL002-F3"
            ]
        )

        self.assertIn(
            "IR-005",
            result["detected_rules"]
        )

        # 최소 1개의 Fact만 제거하면 조합이 깨져야 함
        self.assertEqual(
            len(result["removed_facts"]),
            1
        )

        self.assertEqual(
            len(result["allowed_facts"]),
            1
        )

    # -------------------------------------------------
    # 5. 이미 과거에 발생한 유출을 새 답변 탓으로
    #    잘못 처리하지 않는지 확인
    # -------------------------------------------------

    def test_preexisting_violation(self):

        result = self.controller.evaluate_turn(
            user_id="U3",
            exposed_facts=[
                "OPS001-F2",
                "OPS002-F2"
            ],
            candidate_facts=[
                "PER001-F2"
            ]
        )

        self.assertIn(
            "IR-001",
            result["preexisting_violations"]
        )

        # 이번 Fact는 기존 위반과 무관하므로
        # 불필요하게 제거하면 안 됨
        self.assertEqual(
            result["removed_facts"],
            []
        )

        self.assertEqual(
            result["allowed_facts"],
            ["PER001-F2"]
        )


if __name__ == "__main__":
    unittest.main()
