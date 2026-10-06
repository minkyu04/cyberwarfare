import json
import tempfile
import unittest
from pathlib import Path

from src.dataset_manager import (
    CORE_FILES,
    DatasetManager
)


def write_json(
    path,
    data
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


class TestDatasetManager(
    unittest.TestCase
):

    def setUp(
        self
    ):

        self.temp_dir = (
            tempfile.TemporaryDirectory()
        )

        self.data_dir = Path(
            self.temp_dir.name
        )

        self.full_dir = (
            self.data_dir
            / "full_experiment"
        )

        self.full_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------------------
        # Pilot files
        # ---------------------------------------------

        for filename in CORE_FILES:

            write_json(
                self.data_dir
                / filename,
                {
                    "dataset":
                        "pilot",

                    "file":
                        filename
                }
            )

        # ---------------------------------------------
        # Full files
        # ---------------------------------------------

        write_json(
            self.full_dir
            / "users.json",
            [
                {
                    "user_id":
                        f"U{i}"
                }
                for i in range(
                    12
                )
            ]
        )

        write_json(
            self.full_dir
            / "documents.json",
            [
                {
                    "document_id":
                        f"D{i}"
                }
                for i in range(
                    120
                )
            ]
        )

        write_json(
            self.full_dir
            / "inference_rules.json",
            [
                {
                    "rule_id":
                        f"R{i}"
                }
                for i in range(
                    60
                )
            ]
        )

        write_json(
            self.full_dir
            / "fact_utility.json",
            [
                {
                    "fact_id":
                        f"F{i}"
                }
                for i in range(
                    360
                )
            ]
        )

        write_json(
            self.full_dir
            / "temporary_authorizations.json",
            [
                {
                    "authorization_id":
                        f"TA{i}"
                }
                for i in range(
                    6
                )
            ]
        )

        write_json(
            self.full_dir
            / "policies.json",
            [
                {
                    "policy_id":
                        "P1"
                }
            ]
        )

        self.manager = (
            DatasetManager(
                data_dir=
                    self.data_dir
            )
        )

    def tearDown(
        self
    ):

        self.temp_dir.cleanup()

    # =================================================
    # 1. Validation
    # =================================================

    def test_validate_full_dataset(
        self
    ):

        result = (
            self.manager
            .validate_full_dataset()
        )

        self.assertEqual(
            result[
                "documents"
            ],
            120
        )

        self.assertEqual(
            result[
                "users"
            ],
            12
        )

        self.assertEqual(
            result[
                "rules"
            ],
            60
        )

        self.assertEqual(
            result[
                "utilities"
            ],
            360
        )

        self.assertEqual(
            result[
                "temporary_authorizations"
            ],
            6
        )

    # =================================================
    # 2. Activate
    # =================================================

    def test_activate_full(
        self
    ):

        self.manager.activate_full()

        active_users = (
            json.loads(
                (
                    self.data_dir
                    / "users.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
        )

        self.assertEqual(
            len(
                active_users
            ),
            12
        )

        self.assertEqual(
            self.manager
            .detect_dataset(),
            "full_experiment"
        )

    # =================================================
    # 3. Backup
    # =================================================

    def test_activation_creates_backup(
        self
    ):

        self.manager.activate_full()

        backup_file = (
            self.data_dir
            / ".dataset_backup"
            / "pilot"
            / "users.json"
        )

        self.assertTrue(
            backup_file.exists()
        )

        backup_data = (
            json.loads(
                backup_file.read_text(
                    encoding="utf-8"
                )
            )
        )

        self.assertEqual(
            backup_data[
                "dataset"
            ],
            "pilot"
        )

    # =================================================
    # 4. Restore
    # =================================================

    def test_restore_pilot(
        self
    ):

        self.manager.activate_full()

        self.manager.restore_pilot()

        active_users = (
            json.loads(
                (
                    self.data_dir
                    / "users.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
        )

        self.assertEqual(
            active_users[
                "dataset"
            ],
            "pilot"
        )

        self.assertEqual(
            self.manager
            .detect_dataset(),
            "pilot"
        )

    # =================================================
    # 5. Idempotent Activation
    # =================================================

    def test_activate_full_twice(
        self
    ):

        self.manager.activate_full()

        self.manager.activate_full()

        self.assertEqual(
            self.manager
            .detect_dataset(),
            "full_experiment"
        )


if __name__ == "__main__":

    unittest.main()
