from copy import deepcopy


class RevocationAwareSessionManager:
    """
    대화 Session의 공개이력과 임시권한 의존성을 관리한다.

    핵심 원칙
    --------------------------------------------------
    이미 공개된 Fact 자체를 삭제하거나 회수했다고
    가정하지 않는다.

    대신 어떤 정보조합 Rule이 임시권한에 의해
    허용되었는지를 기록한다.

    권한 만료 후에는 해당 Rule을 현재 정책으로 다시
    평가하고, 실제 Fact 제거는 IncrementalDisclosureController가
    수행한다.
    """

    def __init__(
        self,
        disclosure_controller
    ):

        self.controller = (
            disclosure_controller
        )

        self.sessions = {}

        self.rule_map = {
            rule["rule_id"]:
                rule
            for rule
            in self.controller.rules
        }

    # =================================================
    # Session
    # =================================================

    def get_session(
        self,
        session_id,
        user_id
    ):

        if session_id not in self.sessions:

            self.sessions[
                session_id
            ] = {
                "user_id":
                    user_id,

                "exposure_events":
                    []
            }

        session = self.sessions[
            session_id
        ]

        if (
            session["user_id"]
            != user_id
        ):

            raise ValueError(
                "Session user mismatch: "
                f"{session_id} belongs to "
                f"{session['user_id']}"
            )

        return session

    # =================================================
    # Session 초기화
    # =================================================

    def reset_session(
        self,
        session_id
    ):

        if session_id in self.sessions:

            del self.sessions[
                session_id
            ]

    # =================================================
    # 전체 공개 Fact History
    # =================================================

    def get_exposed_fact_ids(
        self,
        session_id,
        user_id
    ):

        session = self.get_session(
            session_id,
            user_id
        )

        fact_ids = [
            event["fact_id"]
            for event
            in session[
                "exposure_events"
            ]
        ]

        return list(
            dict.fromkeys(
                fact_ids
            )
        )

    # =================================================
    # 임시권한 때문에 허용된 Rule 탐지
    # =================================================

    def find_temporarily_lifted_rules(
        self,
        user_id,
        exposed_facts_before,
        candidate_facts,
        as_of
    ):

        # ---------------------------------------------
        # 임시권한 없음
        # ---------------------------------------------

        baseline_violations = (
            self.controller
            .find_current_response_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts_before,

                candidate_facts=
                    candidate_facts,

                as_of=
                    None
            )
        )

        # ---------------------------------------------
        # 현재 동적권한 적용
        # ---------------------------------------------

        current_violations = (
            self.controller
            .find_current_response_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts_before,

                candidate_facts=
                    candidate_facts,

                as_of=
                    as_of
            )
        )

        baseline_ids = {
            rule["rule_id"]
            for rule
            in baseline_violations
        }

        current_ids = {
            rule["rule_id"]
            for rule
            in current_violations
        }

        return sorted(
            baseline_ids
            - current_ids
        )

    # =================================================
    # 공개 Event 기록
    # =================================================

    def record_disclosure(
        self,
        session_id,
        user_id,
        disclosed_fact_ids,
        exposed_facts_before,
        candidate_facts,
        as_of=None
    ):

        session = self.get_session(
            session_id,
            user_id
        )

        if not disclosed_fact_ids:

            return

        authorization_context = (
            self.controller
            .get_authorization_context(
                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        lifted_rule_ids = (
            self.find_temporarily_lifted_rules(
                user_id=
                    user_id,

                exposed_facts_before=
                    exposed_facts_before,

                candidate_facts=
                    candidate_facts,

                as_of=
                    as_of
            )
        )

        for fact_id in (
            disclosed_fact_ids
        ):

            related_temporary_rules = []

            for rule_id in (
                lifted_rule_ids
            ):

                rule = self.rule_map[
                    rule_id
                ]

                if (
                    fact_id
                    in rule[
                        "required_facts"
                    ]
                ):

                    related_temporary_rules.append(
                        rule_id
                    )

            event = {
                "fact_id":
                    fact_id,

                "disclosed_at":
                    (
                        str(as_of)
                        if as_of is not None
                        else None
                    ),

                "active_authorization_ids":
                    list(
                        authorization_context[
                            "active_authorization_ids"
                        ]
                    ),

                "temporary_rule_ids":
                    related_temporary_rules
            }

            session[
                "exposure_events"
            ].append(
                event
            )

    # =================================================
    # Session에서 사용된 임시 Rule
    # =================================================

    def get_temporary_rule_ids(
        self,
        session_id,
        user_id
    ):

        session = self.get_session(
            session_id,
            user_id
        )

        rule_ids = set()

        for event in (
            session[
                "exposure_events"
            ]
        ):

            rule_ids.update(
                event.get(
                    "temporary_rule_ids",
                    []
                )
            )

        return sorted(
            rule_ids
        )

    # =================================================
    # 현재 만료/회수된 Rule
    # =================================================

    def get_revoked_rule_ids(
        self,
        session_id,
        user_id,
        as_of=None
    ):

        temporary_rule_ids = (
            self.get_temporary_rule_ids(
                session_id,
                user_id
            )
        )

        revoked = []

        for rule_id in (
            temporary_rule_ids
        ):

            rule = self.rule_map[
                rule_id
            ]

            allowed_now = (
                self.controller
                .can_receive_combination(
                    user_id=
                        user_id,

                    rule=
                        rule,

                    as_of=
                        as_of
                )
            )

            if not allowed_now:

                revoked.append(
                    rule_id
                )

        return sorted(
            revoked
        )

    # =================================================
    # Snapshot
    # =================================================

    def get_session_snapshot(
        self,
        session_id,
        user_id,
        as_of=None
    ):

        session = self.get_session(
            session_id,
            user_id
        )

        return {
            "session_id":
                session_id,

            "user_id":
                user_id,

            "exposed_fact_ids":
                self.get_exposed_fact_ids(
                    session_id,
                    user_id
                ),

            "temporary_rule_ids":
                self.get_temporary_rule_ids(
                    session_id,
                    user_id
                ),

            "revoked_rule_ids":
                self.get_revoked_rule_ids(
                    session_id,
                    user_id,
                    as_of
                ),

            "exposure_events":
                deepcopy(
                    session[
                        "exposure_events"
                    ]
                )
        }
