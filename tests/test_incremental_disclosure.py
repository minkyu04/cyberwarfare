import unittest

from src.incremental_disclosure import (
    IncrementalDisclosureController
)


class TestIncrementalDisclosureController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        모든 테스트에서 동일한 Controller를 사용한다.
        """
        cls.controller = IncrementalDisclosureController()

    # =================================================
    # 1. 다중턴 누적 정보결합 위반 탐지
    # =================================================

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

        # OPS001-F2가 이미 공개된 상태에서
        # OPS002-F2를 추가하면 IR-001이 완성된다.
        self.assertIn(
            "IR-001",
            result["detected_rules"]
        )

        # 이번 답변에서 제거 가능한 Fact는
        # OPS002-F2 하나뿐이다.
        self.assertEqual(
            result["removed_facts"],
            [
                "OPS002-F2"
            ]
        )

        self.assertEqual(
            result["allowed_facts"],
            []
        )

        # 제거된 Fact의 업무가치
        self.assertEqual(
            result["removed_utility"],
            2.0
        )

        self.assertEqual(
            result["retained_utility"],
            0.0
        )

        self.assertEqual(
            result["utility_retention_rate"],
            0.0
        )

    # =================================================
    # 2. 가중 최소 업무손실 제거 알고리즘
    # =================================================

    def test_weighted_minimum_loss_removal(self):

        result = self.controller.evaluate_turn(
            user_id="U3",
            exposed_facts=[],
            candidate_facts=[
                "OPS001-F2",
                "OPS002-F2",
                "OPS002-F3"
            ]
        )

        # 두 개의 금지 정보조합이 동시에 형성된다.
        self.assertCountEqual(
            result["detected_rules"],
            [
                "IR-001",
                "IR-002"
            ]
        )

        # 업무가치
        #
        # OPS001-F2 = 8
        # OPS002-F2 = 2
        # OPS002-F3 = 1
        #
        # 기존 최소 개수 제거:
        # OPS001-F2 하나 제거
        # 손실 = 8
        #
        # 가중 최소손실 제거:
        # OPS002-F2 + OPS002-F3 제거
        # 손실 = 3
        #
        # 따라서 새로운 알고리즘은 Fact 수가
        # 더 많더라도 총 업무손실이 적은 조합을
        # 선택해야 한다.

        self.assertCountEqual(
            result["removed_facts"],
            [
                "OPS002-F2",
                "OPS002-F3"
            ]
        )

        self.assertEqual(
            result["allowed_facts"],
            [
                "OPS001-F2"
            ]
        )

        self.assertEqual(
            result["candidate_utility"],
            11.0
        )

        self.assertEqual(
            result["removed_utility"],
            3.0
        )

        self.assertEqual(
            result["retained_utility"],
            8.0
        )

        self.assertAlmostEqual(
            result["utility_retention_rate"],
            8 / 11
        )

    # =================================================
    # 3. 정상 권한 사용자의 정보는 보존
    # =================================================

    def test_authorized_combination_is_preserved(self):

        result = self.controller.evaluate_turn(
            user_id="U4",
            exposed_facts=[],
            candidate_facts=[
                "LOG001-F3",
                "LOG002-F3"
            ]
        )

        # U4:
        # clearance = 3
        # mission = CYBER
        #
        # IR-004 요구조건을 충족하므로
        # 정보결합을 허용해야 한다.

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

        self.assertEqual(
            result["candidate_utility"],
            6.0
        )

        self.assertEqual(
            result["removed_utility"],
            0.0
        )

        self.assertEqual(
            result["retained_utility"],
            6.0
        )

        self.assertEqual(
            result["utility_retention_rate"],
            1.0
        )

    # =================================================
    # 4. Clearance가 충분해도 Mission이 다르면 제한
    # =================================================

    def test_mission_mismatch(self):

        result = self.controller.evaluate_turn(
            user_id="U4",
            exposed_facts=[],
            candidate_facts=[
                "VUL001-F3",
                "VUL002-F3"
            ]
        )

        # IR-005
        #
        # required_clearance = 3
        # allowed_missions = CYBER_COMMAND
        #
        # U4의 clearance는 3이지만
        # mission은 CYBER이므로 조합 권한이 없다.

        self.assertIn(
            "IR-005",
            result["detected_rules"]
        )

        # 업무가치
        #
        # VUL001-F3 = 5
        # VUL002-F3 = 3
        #
        # 둘 중 하나만 제거하면 IR-005가 깨지므로
        # 업무손실이 더 작은 VUL002-F3을 제거해야 한다.

        self.assertEqual(
            result["removed_facts"],
            [
                "VUL002-F3"
            ]
        )

        self.assertEqual(
            result["allowed_facts"],
            [
                "VUL001-F3"
            ]
        )

        self.assertEqual(
            result["candidate_utility"],
            8.0
        )

        self.assertEqual(
            result["removed_utility"],
            3.0
        )

        self.assertEqual(
            result["retained_utility"],
            5.0
        )

        self.assertAlmostEqual(
            result["utility_retention_rate"],
            5 / 8
        )

    # =================================================
    # 5. 이전 턴에서 이미 발생한 위반과
    #    현재 답변을 구분
    # =================================================

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

        # IR-001은 이미 과거 공개정보만으로
        # 완성된 상태이다.
        self.assertIn(
            "IR-001",
            result["preexisting_violations"]
        )

        # 이번 PER001-F2는 IR-001과 무관하므로
        # 과거에 발생한 위반 때문에 이번 정상정보를
        # 불필요하게 제거하면 안 된다.
        self.assertEqual(
            result["removed_facts"],
            []
        )

        self.assertEqual(
            result["allowed_facts"],
            [
                "PER001-F2"
            ]
        )

        self.assertEqual(
            result["candidate_utility"],
            4.0
        )

        self.assertEqual(
            result["retained_utility"],
            4.0
        )

        self.assertEqual(
            result["utility_retention_rate"],
            1.0
        )

    # =================================================
    # 6. 동일 업무가치일 때 결과가 결정론적인지 검사
    # =================================================

    def test_equal_utility_tie_breaking(self):

        result = self.controller.evaluate_turn(
            user_id="U2",
            exposed_facts=[],
            candidate_facts=[
                "PER001-F2",
                "PER002-F2"
            ]
        )

        # U2는 ADMIN 임무지만 clearance=1이다.
        # IR-003은 clearance=2가 필요하므로 위반한다.

        self.assertIn(
            "IR-003",
            result["detected_rules"]
        )

        # 두 Fact의 business_value는 모두 4.
        # 동일 손실이라면 결정론적 tie-breaking에 따라
        # PER001-F2가 선택되어야 한다.

        self.assertEqual(
            result["removed_facts"],
            [
                "PER001-F2"
            ]
        )

        self.assertEqual(
            result["allowed_facts"],
            [
                "PER002-F2"
            ]
        )

        self.assertEqual(
            result["removed_utility"],
            4.0
        )

        self.assertEqual(
            result["retained_utility"],
            4.0
        )


if __name__ == "__main__":
    unittest.main()
