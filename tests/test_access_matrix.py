import unittest

from src.policy_engine import is_allowed


class TestAccessMatrix(unittest.TestCase):

    def test_full_access_matrix(self):

        expected_access = {

            "U1": {
                "PUB-001": True,
                "PUB-002": True,
                "PER-001": False,
                "PER-002": False,
                "OPS-001": False,
                "OPS-002": False,
                "LOG-001": False,
                "LOG-002": False,
                "VUL-001": False,
                "VUL-002": False
            },

            "U2": {
                "PUB-001": True,
                "PUB-002": True,
                "PER-001": True,
                "PER-002": True,
                "OPS-001": False,
                "OPS-002": False,
                "LOG-001": False,
                "LOG-002": False,
                "VUL-001": False,
                "VUL-002": False
            },

            "U3": {
                "PUB-001": True,
                "PUB-002": True,
                "PER-001": False,
                "PER-002": False,
                "OPS-001": True,
                "OPS-002": True,
                "LOG-001": False,
                "LOG-002": False,
                "VUL-001": False,
                "VUL-002": False
            },

            "U4": {
                "PUB-001": True,
                "PUB-002": True,
                "PER-001": False,
                "PER-002": False,
                "OPS-001": False,
                "OPS-002": False,
                "LOG-001": True,
                "LOG-002": True,
                "VUL-001": True,
                "VUL-002": True
            }
        }

        checked = 0

        for user_id, documents in expected_access.items():

            for document_id, expected in documents.items():

                actual = is_allowed(user_id, document_id)

                self.assertEqual(
                    actual,
                    expected,
                    msg=(
                        f"{user_id} -> {document_id}: "
                        f"expected={expected}, actual={actual}"
                    )
                )

                checked += 1

        self.assertEqual(checked, 40)


if __name__ == "__main__":
    unittest.main()
