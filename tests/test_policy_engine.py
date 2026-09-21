import unittest

from src.policy_engine import is_allowed


class TestPolicyEngine(unittest.TestCase):

    def test_general_user_cannot_access_vulnerability_report(self):
        self.assertFalse(
            is_allowed("U1", "VUL-001")
        )

    def test_personnel_staff_cannot_access_vulnerability_report(self):
        self.assertFalse(
            is_allowed("U2", "VUL-001")
        )

    def test_operations_staff_cannot_access_vulnerability_report(self):
        self.assertFalse(
            is_allowed("U3", "VUL-001")
        )

    def test_security_admin_can_access_vulnerability_report(self):
        self.assertTrue(
            is_allowed("U4", "VUL-001")
        )


if __name__ == "__main__":
    unittest.main()
