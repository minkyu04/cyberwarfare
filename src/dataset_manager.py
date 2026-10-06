import argparse
import hashlib
import json
import shutil
from pathlib import Path


BASE_DIR = Path(
    __file__
).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
)

FULL_DATA_DIR = (
    DATA_DIR
    / "full_experiment"
)

BACKUP_DIR = (
    DATA_DIR
    / ".dataset_backup"
    / "pilot"
)

STATE_FILE = (
    DATA_DIR
    / ".active_dataset.json"
)


CORE_FILES = [
    "users.json",
    "policies.json",
    "documents.json",
    "inference_rules.json",
    "fact_utility.json",
    "temporary_authorizations.json"
]


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(
    path,
    data
):

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


class DatasetManager:

    def __init__(
        self,
        data_dir=DATA_DIR
    ):

        self.data_dir = Path(
            data_dir
        )

        self.full_data_dir = (
            self.data_dir
            / "full_experiment"
        )

        self.backup_dir = (
            self.data_dir
            / ".dataset_backup"
            / "pilot"
        )

        self.state_file = (
            self.data_dir
            / ".active_dataset.json"
        )

    # =====================================================
    # Validation
    # =====================================================

    def validate_active_files(
        self
    ):

        missing = []

        for filename in CORE_FILES:

            path = (
                self.data_dir
                / filename
            )

            if not path.exists():

                missing.append(
                    filename
                )

        if missing:

            raise FileNotFoundError(
                "현재 data/에 필요한 파일이 없습니다: "
                + ", ".join(
                    missing
                )
            )

    def validate_full_dataset(
        self
    ):

        missing = []

        for filename in CORE_FILES:

            path = (
                self.full_data_dir
                / filename
            )

            if not path.exists():

                missing.append(
                    filename
                )

        if missing:

            raise FileNotFoundError(
                "full_experiment에 필요한 파일이 없습니다: "
                + ", ".join(
                    missing
                )
            )

        documents = load_json(
            self.full_data_dir
            / "documents.json"
        )

        users = load_json(
            self.full_data_dir
            / "users.json"
        )

        rules = load_json(
            self.full_data_dir
            / "inference_rules.json"
        )

        utilities = load_json(
            self.full_data_dir
            / "fact_utility.json"
        )

        temporary_authorizations = load_json(
            self.full_data_dir
            / "temporary_authorizations.json"
        )

        if len(documents) != 120:

            raise ValueError(
                "Full dataset documents != 120"
            )

        if len(users) != 12:

            raise ValueError(
                "Full dataset users != 12"
            )

        if len(rules) != 60:

            raise ValueError(
                "Full dataset inference rules != 60"
            )

        if len(utilities) != 360:

            raise ValueError(
                "Full dataset utility facts != 360"
            )

        if len(
            temporary_authorizations
        ) != 6:

            raise ValueError(
                "Full dataset temporary authorizations != 6"
            )

        return {
            "documents":
                len(documents),

            "users":
                len(users),

            "rules":
                len(rules),

            "utilities":
                len(utilities),

            "temporary_authorizations":
                len(
                    temporary_authorizations
                )
        }

    # =====================================================
    # Backup
    # =====================================================

    def backup_pilot_dataset(
        self
    ):

        self.validate_active_files()

        self.backup_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for filename in CORE_FILES:

            source = (
                self.data_dir
                / filename
            )

            destination = (
                self.backup_dir
                / filename
            )

            if not destination.exists():

                shutil.copy2(
                    source,
                    destination
                )

    # =====================================================
    # State
    # =====================================================

    def read_state(
        self
    ):

        if not self.state_file.exists():

            return {
                "active_dataset":
                    "unknown"
            }

        return load_json(
            self.state_file
        )

    def write_state(
        self,
        dataset_name
    ):

        state = {
            "active_dataset":
                dataset_name,

            "core_files":
                CORE_FILES
        }

        save_json(
            self.state_file,
            state
        )

    # =====================================================
    # Activate Full Dataset
    # =====================================================

    def activate_full(
        self
    ):

        validation = (
            self.validate_full_dataset()
        )

        state = (
            self.read_state()
        )

        if (
            state.get(
                "active_dataset"
            )
            == "full_experiment"
        ):

            print(
                "Full experiment dataset is already active."
            )

            return validation

        self.backup_pilot_dataset()

        for filename in CORE_FILES:

            source = (
                self.full_data_dir
                / filename
            )

            destination = (
                self.data_dir
                / filename
            )

            shutil.copy2(
                source,
                destination
            )

        self.write_state(
            "full_experiment"
        )

        print(
            "Activated full experiment dataset."
        )

        return validation

    # =====================================================
    # Restore Pilot Dataset
    # =====================================================

    def restore_pilot(
        self
    ):

        missing = []

        for filename in CORE_FILES:

            path = (
                self.backup_dir
                / filename
            )

            if not path.exists():

                missing.append(
                    filename
                )

        if missing:

            raise FileNotFoundError(
                "Pilot backup files are missing: "
                + ", ".join(
                    missing
                )
            )

        for filename in CORE_FILES:

            source = (
                self.backup_dir
                / filename
            )

            destination = (
                self.data_dir
                / filename
            )

            shutil.copy2(
                source,
                destination
            )

        self.write_state(
            "pilot"
        )

        print(
            "Restored pilot dataset."
        )

    # =====================================================
    # Dataset Comparison
    # =====================================================

    def files_match(
        self,
        source_dir
    ):

        for filename in CORE_FILES:

            active = (
                self.data_dir
                / filename
            )

            reference = (
                source_dir
                / filename
            )

            if (
                not active.exists()
                or not reference.exists()
            ):

                return False

            if (
                sha256_file(
                    active
                )
                != sha256_file(
                    reference
                )
            ):

                return False

        return True

    def detect_dataset(
        self
    ):

        if (
            self.full_data_dir.exists()
            and self.files_match(
                self.full_data_dir
            )
        ):

            return (
                "full_experiment"
            )

        if (
            self.backup_dir.exists()
            and self.files_match(
                self.backup_dir
            )
        ):

            return "pilot"

        return "mixed_or_unknown"

    # =====================================================
    # Status
    # =====================================================

    def status(
        self
    ):

        detected = (
            self.detect_dataset()
        )

        state = (
            self.read_state()
        )

        result = {
            "recorded_state":
                state.get(
                    "active_dataset",
                    "unknown"
                ),

            "detected_dataset":
                detected,

            "pilot_backup_exists":
                self.backup_dir.exists(),

            "full_dataset_exists":
                self.full_data_dir.exists()
        }

        return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Switch between pilot and "
            "full experiment datasets."
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "status",
            "activate-full",
            "restore-pilot"
        ]
    )

    args = parser.parse_args()

    manager = (
        DatasetManager()
    )

    if args.command == "status":

        print(
            json.dumps(
                manager.status(),
                ensure_ascii=False,
                indent=2
            )
        )

    elif (
        args.command
        == "activate-full"
    ):

        result = (
            manager.activate_full()
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        print(
            json.dumps(
                manager.status(),
                ensure_ascii=False,
                indent=2
            )
        )

    elif (
        args.command
        == "restore-pilot"
    ):

        manager.restore_pilot()

        print(
            json.dumps(
                manager.status(),
                ensure_ascii=False,
                indent=2
            )
        )


if __name__ == "__main__":

    main()
