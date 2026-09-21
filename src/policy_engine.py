import json
from pathlib import Path


# 프로젝트 최상위 폴더
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_json(filename):
    """data 폴더의 JSON 파일을 불러온다."""
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


# 데이터 불러오기
users = load_json("users.json")
documents = load_json("documents.json")
policies = load_json("policies.json")


def get_user(user_id):
    """user_id에 해당하는 사용자 정보를 반환한다."""
    for user in users:
        if user["user_id"] == user_id:
            return user

    raise ValueError(f"사용자를 찾을 수 없습니다: {user_id}")


def get_document(document_id):
    """document_id에 해당하는 문서를 반환한다."""
    for document in documents:
        if document["document_id"] == document_id:
            return document

    raise ValueError(f"문서를 찾을 수 없습니다: {document_id}")


def get_policy(document_type):
    """문서 유형에 적용되는 접근통제 정책을 반환한다."""
    for policy in policies:
        if policy["document_type"] == document_type:
            return policy

    raise ValueError(f"정책을 찾을 수 없습니다: {document_type}")


def matches(value, allowed_values):
    """
    '*'가 포함되어 있으면 모든 값을 허용한다.
    그렇지 않으면 허용 목록에 값이 존재하는지 검사한다.
    """
    if "*" in allowed_values:
        return True

    return value in allowed_values


def is_allowed(user_id, document_id):
    """
    사용자와 문서 정보를 기준으로 접근 허용 여부를 판단한다.

    검사 순서:
    1. clearance
    2. role
    3. department
    4. mission
    """

    user = get_user(user_id)
    document = get_document(document_id)
    policy = get_policy(document["document_type"])

   # 1. 최소 인가수준 및 문서 등급 검사
    required_clearance = max(
        policy["min_clearance"],
        document["classification"]
    )
    
    if user["clearance"] < required_clearance:
    return False

    # 2. 역할 검사
    if not matches(user["role"], policy["allowed_roles"]):
        return False

    # 3. 부서 검사
    if not matches(
        user["department"],
        policy["allowed_departments"]
    ):
        return False

    # 4. 임무 일치 여부 검사
    if policy["enforce_mission_match"]:
        if user["mission"] != document["mission"]:
            return False

    return True


if __name__ == "__main__":

    print("=== Access Control Test ===")

    test_cases = [
        ("U1", "VUL-001"),
        ("U4", "VUL-001")
    ]

    for user_id, document_id in test_cases:
        result = is_allowed(user_id, document_id)

        print(
            f"{user_id} -> {document_id}: "
            f"{'ALLOW' if result else 'DENY'}"
        )
