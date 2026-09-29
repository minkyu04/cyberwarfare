import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_json(filename):

    with open(
        DATA_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


class DynamicAuthorizationManager:
    """
    합성 임시 권한의 유효시간을 검사하고
    사용자의 현재 Effective Authorization을 계산한다.

    기본 원칙

    1. 기본 users.json 권한은 유지한다.
    2. 활성화된 임시 권한만 추가 적용한다.
    3. valid_from 이상부터 활성화한다.
    4. valid_until 시각부터는 만료된 것으로 처리한다.
    5. 임시 권한이 기본 권한을 낮추지는 않는다.
    """

    def __init__(self):

        self.users = load_json(
            "users.json"
        )

        self.authorizations = load_json(
            "temporary_authorizations.json"
        )

        self.user_map = {
            user["user_id"]: user
            for user in self.users
        }

        self.validate_authorizations()

    # =================================================
    # ISO 시간 파싱
    # =================================================

    def parse_datetime(
        self,
        value
    ):

        if isinstance(
            value,
            datetime
        ):

            parsed = value

        elif isinstance(
            value,
            str
        ):

            parsed = (
                datetime.fromisoformat(
                    value
                )
            )

        else:

            raise TypeError(
                "시간 값은 datetime 또는 "
                "ISO 8601 문자열이어야 합니다."
            )

        if parsed.tzinfo is None:

            raise ValueError(
                "시간에는 timezone 정보가 "
                "포함되어야 합니다."
            )

        return parsed

    # =================================================
    # 임시 권한 파일 검증
    # =================================================

    def validate_authorizations(self):

        seen_ids = set()

        for authorization in (
            self.authorizations
        ):

            authorization_id = (
                authorization[
                    "authorization_id"
                ]
            )

            if authorization_id in seen_ids:

                raise ValueError(
                    "중복 authorization_id: "
                    f"{authorization_id}"
                )

            seen_ids.add(
                authorization_id
            )

            user_id = (
                authorization[
                    "user_id"
                ]
            )

            if user_id not in self.user_map:

                raise ValueError(
                    "존재하지 않는 사용자에 대한 "
                    "임시 권한입니다: "
                    f"{user_id}"
                )

            valid_from = (
                self.parse_datetime(
                    authorization[
                        "valid_from"
                    ]
                )
            )

            valid_until = (
                self.parse_datetime(
                    authorization[
                        "valid_until"
                    ]
                )
            )

            if valid_until <= valid_from:

                raise ValueError(
                    f"{authorization_id}: "
                    "valid_until은 valid_from보다 "
                    "뒤여야 합니다."
                )

            temporary_clearance = (
                authorization.get(
                    "temporary_clearance"
                )
            )

            if (
                temporary_clearance
                is not None
                and temporary_clearance < 0
            ):

                raise ValueError(
                    f"{authorization_id}: "
                    "temporary_clearance는 "
                    "0 이상이어야 합니다."
                )

    # =================================================
    # 기본 사용자
    # =================================================

    def get_user(
        self,
        user_id
    ):

        if user_id not in self.user_map:

            raise ValueError(
                f"Unknown user: {user_id}"
            )

        return self.user_map[
            user_id
        ]

    # =================================================
    # 특정 임시 권한 활성 여부
    # =================================================

    def is_authorization_active(
        self,
        authorization,
        as_of
    ):

        if not authorization.get(
            "enabled",
            True
        ):

            return False

        current_time = (
            self.parse_datetime(
                as_of
            )
        )

        valid_from = (
            self.parse_datetime(
                authorization[
                    "valid_from"
                ]
            )
        )

        valid_until = (
            self.parse_datetime(
                authorization[
                    "valid_until"
                ]
            )
        )

        # 시작시각 포함
        # 종료시각 미포함
        return (
            valid_from
            <= current_time
            < valid_until
        )

    # =================================================
    # 활성 임시 권한 조회
    # =================================================

    def get_active_authorizations(
        self,
        user_id,
        as_of
    ):

        self.get_user(
            user_id
        )

        active = []

        for authorization in (
            self.authorizations
        ):

            if (
                authorization[
                    "user_id"
                ]
                != user_id
            ):

                continue

            if self.is_authorization_active(
                authorization,
                as_of
            ):

                active.append(
                    authorization
                )

        return active

    # =================================================
    # Effective Authorization 계산
    # =================================================

    def get_effective_context(
        self,
        user_id,
        as_of=None
    ):

        user = self.get_user(
            user_id
        )

        base_clearance = int(
            user["clearance"]
        )

        effective_clearance = (
            base_clearance
        )

        missions = set()

        base_mission = user.get(
            "mission"
        )

        if (
            base_mission
            and base_mission != "NONE"
        ):

            missions.add(
                base_mission
            )

        active_authorizations = []

        # as_of=None이면 기존 실험과의 호환을 위해
        # 임시 권한을 적용하지 않는다.
        if as_of is not None:

            active_authorizations = (
                self.get_active_authorizations(
                    user_id,
                    as_of
                )
            )

            for authorization in (
                active_authorizations
            ):

                temp_clearance = (
                    authorization.get(
                        "temporary_clearance"
                    )
                )

                if (
                    temp_clearance
                    is not None
                ):

                    effective_clearance = max(
                        effective_clearance,
                        int(
                            temp_clearance
                        )
                    )

                for mission in (
                    authorization.get(
                        "temporary_missions",
                        []
                    )
                ):

                    missions.add(
                        mission
                    )

        return {
            "user_id":
                user_id,

            "base_clearance":
                base_clearance,

            "effective_clearance":
                effective_clearance,

            "base_mission":
                base_mission,

            "effective_missions":
                sorted(
                    missions
                ),

            "active_authorization_ids":
                [
                    authorization[
                        "authorization_id"
                    ]
                    for authorization
                    in active_authorizations
                ],

            "as_of":
                (
                    str(as_of)
                    if as_of is not None
                    else None
                )
        }
