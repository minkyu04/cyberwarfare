import json
from pathlib import Path


# ============================================================
# Path
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
)


# ============================================================
# Data Loading
# ============================================================

def load_json(
    filename
):
    """
    현재 활성화된 data/ 폴더의 JSON 파일을 불러온다.
    """

    path = (
        DATA_DIR
        / filename
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


users = []
documents = []
policies = []


def reload_data():
    """
    data/의 users, documents, policies를 다시 읽는다.

    DatasetManager로 pilot/full_experiment를 전환한 뒤
    같은 Python 세션에서 정책 데이터를 갱신할 때 사용할 수 있다.
    """

    global users
    global documents
    global policies

    users = load_json(
        "users.json"
    )

    documents = load_json(
        "documents.json"
    )

    policies = load_json(
        "policies.json"
    )

    return {
        "users":
            len(users),

        "documents":
            len(documents),

        "policies":
            len(policies)
    }


# 모듈 최초 import 시 현재 활성 데이터 로드
reload_data()


# ============================================================
# Lookup
# ============================================================

def get_user(
    user_id
):
    """
    user_id에 해당하는 사용자 정보를 반환한다.
    """

    for user in users:

        if (
            user["user_id"]
            == user_id
        ):

            return user

    raise ValueError(
        f"사용자를 찾을 수 없습니다: "
        f"{user_id}"
    )


def get_document(
    document_id
):
    """
    document_id에 해당하는 문서를 반환한다.
    """

    for document in documents:

        if (
            document["document_id"]
            == document_id
        ):

            return document

    raise ValueError(
        f"문서를 찾을 수 없습니다: "
        f"{document_id}"
    )


def get_policy(
    document_type
):
    """
    문서 유형에 적용되는 접근통제 정책을 반환한다.
    """

    for policy in policies:

        if (
            policy["document_type"]
            == document_type
        ):

            return policy

    raise ValueError(
        f"정책을 찾을 수 없습니다: "
        f"{document_type}"
    )


# ============================================================
# Policy Helpers
# ============================================================

def matches(
    value,
    allowed_values
):
    """
    '*'가 포함되어 있으면 모든 값을 허용한다.
    그렇지 않으면 허용 목록에 값이 존재하는지 검사한다.
    """

    if (
        "*"
        in allowed_values
    ):

        return True

    return (
        value
        in allowed_values
    )


def requires_mission_match(
    policy
):
    """
    기존 pilot 데이터와 full_experiment 데이터의
    정책 스키마 차이를 모두 지원한다.

    지원 키:
    1. enforce_mission_match
    2. require_mission
    3. mission_required

    둘 이상의 키가 동시에 존재하면서 값이 서로 다르면
    정책 정의 오류로 처리한다.
    """

    supported_keys = [
        "enforce_mission_match",
        "require_mission",
        "mission_required"
    ]

    found = []

    for key in supported_keys:

        if key in policy:

            found.append(
                (
                    key,
                    bool(
                        policy[key]
                    )
                )
            )

    # 어떤 키도 없으면 mission 제약 없음
    if not found:

        return False

    values = {
        value
        for _, value
        in found
    }

    if (
        len(values)
        > 1
    ):

        raise ValueError(
            "Mission policy 설정이 서로 충돌합니다: "
            f"{policy}"
        )

    return found[0][1]


def get_required_clearance(
    policy,
    document
):
    """
    정책 최소 인가수준과 문서 자체 등급 중
    더 높은 값을 요구한다.
    """

    policy_clearance = int(
        policy.get(
            "min_clearance",
            0
        )
    )

    document_clearance = int(
        document.get(
            "classification",
            0
        )
    )

    return max(
        policy_clearance,
        document_clearance
    )


# ============================================================
# Access Decision
# ============================================================

def is_allowed(
    user_id,
    document_id
):
    """
    사용자와 문서 정보를 기준으로
    정적 접근 허용 여부를 판단한다.

    검사 순서:
    1. clearance
    2. role
    3. department
    4. mission

    주의:
    이 함수는 기본 정적 권한을 판정한다.
    임시권한 및 시간 기반 권한 변경은
    DynamicAuthorizationManager에서 별도로 처리한다.
    """

    user = get_user(
        user_id
    )

    document = get_document(
        document_id
    )

    policy = get_policy(
        document[
            "document_type"
        ]
    )

    # --------------------------------------------------------
    # 1. Clearance
    # --------------------------------------------------------

    required_clearance = (
        get_required_clearance(
            policy,
            document
        )
    )

    user_clearance = int(
        user.get(
            "clearance",
            0
        )
    )

    if (
        user_clearance
        < required_clearance
    ):

        return False

    # --------------------------------------------------------
    # 2. Role
    # --------------------------------------------------------

    allowed_roles = (
        policy.get(
            "allowed_roles",
            []
        )
    )

    if not matches(
        user.get(
            "role"
        ),
        allowed_roles
    ):

        return False

    # --------------------------------------------------------
    # 3. Department
    # --------------------------------------------------------

    allowed_departments = (
        policy.get(
            "allowed_departments",
            []
        )
    )

    if not matches(
        user.get(
            "department"
        ),
        allowed_departments
    ):

        return False

    # --------------------------------------------------------
    # 4. Mission
    # --------------------------------------------------------

    if requires_mission_match(
        policy
    ):

        user_mission = (
            user.get(
                "mission"
            )
        )

        document_mission = (
            document.get(
                "mission"
            )
        )

        if (
            user_mission
            != document_mission
        ):

            return False

    return True


# ============================================================
# Utility
# ============================================================

def get_allowed_document_ids(
    user_id
):
    """
    특정 사용자가 정적으로 접근 가능한
    전체 문서 ID를 반환한다.
    """

    allowed = []

    for document in documents:

        document_id = (
            document[
                "document_id"
            ]
        )

        if is_allowed(
            user_id,
            document_id
        ):

            allowed.append(
                document_id
            )

    return allowed


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    print(
        "=== Policy Engine ==="
    )

    print(
        "Users:",
        len(users)
    )

    print(
        "Documents:",
        len(documents)
    )

    print(
        "Policies:",
        len(policies)
    )

    user_ids = {
        user[
            "user_id"
        ]
        for user in users
    }

    document_ids = {
        document[
            "document_id"
        ]
        for document in documents
    }

    # --------------------------------------------------------
    # Pilot dataset
    # --------------------------------------------------------

    if (
        "U1" in user_ids
        and "U4" in user_ids
    ):

        test_cases = [
            (
                "U1",
                "PUB-001"
            ),
            (
                "U1",
                "VUL-001"
            ),
            (
                "U4",
                "VUL-001"
            )
        ]

    # --------------------------------------------------------
    # Full experiment dataset
    # --------------------------------------------------------

    else:

        test_cases = [
            (
                "G01",
                "PUB-001"
            ),
            (
                "P01",
                "PER-001"
            ),
            (
                "O01",
                "OPS-001"
            ),
            (
                "S01",
                "LOG-001"
            ),
            (
                "S01",
                "VUL-001"
            )
        ]

    print()

    for (
        user_id,
        document_id
    ) in test_cases:

        if (
            user_id
            not in user_ids
            or document_id
            not in document_ids
        ):

            continue

        result = is_allowed(
            user_id,
            document_id
        )

        print(
            f"{user_id} -> "
            f"{document_id}: "
            f"{'ALLOW' if result else 'DENY'}"
        )

    print(
        "\n=== Allowed Document Counts ==="
    )

    for user_id in sorted(
        user_ids
    ):

        allowed = (
            get_allowed_document_ids(
                user_id
            )
        )

        print(
            f"{user_id}: "
            f"{len(allowed)}"
        )
