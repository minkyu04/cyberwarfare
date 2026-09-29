from copy import deepcopy


class RevocationAwareSessionManager:
    """
    대화 세션의 정보 공개 이력을 관리한다.

    핵심 원칙
    --------------------------------------------------
    1. 이미 사용자에게 공개된 Fact는 Exposure History에서
       삭제하지 않는다.

    2. 임시 권한에 의존하여 공개된 Fact는 해당 권한이
       만료된 후 AI가 다시 재사용하지 못하도록 표시한다.

    3. 과거 공개 Fact는 이후 정보결합 위험 계산에는
       계속 사용한다.

    즉,
        Human knowledge != AI reusable session context

    를 구분한다.
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
    # Session 생성 / 조회
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
    # 특정 Turn에서 임시권한 덕분에 해제된 Rule 탐지
    # =================================================

    def find_temporarily_lifted_rules(
        self,
        user_id,
        exposed_facts_before,
        candidate_facts,
        as_of
    ):

        # ---------------------------------------------
        # 임시 권한을 전혀 적용하지 않았을 때
        # ---------------------------------------------

        baseline_violations = (
            self.controller.find_violations(
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
        # 현재 시각의 동적 권한 적용
        # ---------------------------------------------

        current_violations = (
            self.controller.find_violations(
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

        # baseline에서는 위반인데
        # 현재 임시권한 때문에 위반이 아니게 된 Rule
        lifted_rule_ids = sorted(
            baseline_ids
            - current_ids
        )

        return lifted_rule_ids

    # =================================================
    # Fact 공개 기록
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

        for fact_id in disclosed_fact_ids:

            temporary_rule_ids = []

            for rule_id in lifted_rule_ids:

                rule = self.rule_map[
                    rule_id
                ]

                if (
                    fact_id
                    in rule[
                        "required_facts"
                    ]
                ):

                    temporary_rule_ids.append(
                        rule_id
                    )

            event = {
                "fact_id":
                    fact_id,

                "disclosed_at":
                    (
                        str(as_of)
                        if as_of
                        is not None
                        else None
                    ),

                "active_authorization_ids":
                    list(
                        authorization_context[
                            "active_authorization_ids"
                        ]
                    ),

                "temporary_rule_ids":
                    temporary_rule_ids
            }

            session[
                "exposure_events"
            ].append(
                event
            )

    # =================================================
    # 과거 공개 Event가 현재도 재사용 가능한지 판단
    # =================================================

    def is_event_reusable(
        self,
        user_id,
        event,
        as_of=None
    ):

        temporary_rule_ids = (
            event[
                "temporary_rule_ids"
            ]
        )

        # ---------------------------------------------
        # 임시 권한에 의존하지 않고 공개된 Fact
        # ---------------------------------------------

        if not temporary_rule_ids:

            return True

        # ---------------------------------------------
        # 임시권한 의존 Fact
        #
        # 관련 Rule이 현재도 모두 허용되는지 확인한다.
        # ---------------------------------------------

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

                return False

        return True

    # =================================================
    # 현재 회수된 Fact 계산
    # =================================================

    def get_revoked_fact_ids(
        self,
        session_id,
        user_id,
        as_of=None
    ):

        session = self.get_session(
            session_id,
            user_id
        )

        # ---------------------------------------------
        # 같은 Fact가 여러 번 공개됐을 수도 있다.
        #
        # 한 번이라도 비임시권한 상태에서 정당하게
        # 공개된 기록이 있다면 해당 Fact 자체를
        # Session reuse 차원에서 회수하지 않는다.
        # ---------------------------------------------

        events_by_fact = {}

        for event in (
            session[
                "exposure_events"
            ]
        ):

            fact_id = event[
                "fact_id"
            ]

            events_by_fact.setdefault(
                fact_id,
                []
            )

            events_by_fact[
                fact_id
            ].append(
                event
            )

        revoked = []

        for (
            fact_id,
            events
        ) in events_by_fact.items():

            reusable_event_exists = any(
                self.is_event_reusable(
                    user_id=
                        user_id,

                    event=
                        event,

                    as_of=
                        as_of
                )

                for event
                in events
            )

            if not reusable_event_exists:

                revoked.append(
                    fact_id
                )

        return sorted(
            revoked
        )

    # =================================================
    # 현재 Candidate 중 회수된 Fact 차단
    # =================================================

    def filter_revoked_candidates(
        self,
        session_id,
        user_id,
        candidate_fact_ids,
        as_of=None
    ):

        revoked = set(
            self.get_revoked_fact_ids(
                session_id=
                    session_id,

                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        blocked = [
            fact_id
            for fact_id
            in candidate_fact_ids
            if fact_id in revoked
        ]

        reusable = [
            fact_id
            for fact_id
            in candidate_fact_ids
            if fact_id not in revoked
        ]

        return {
            "reusable_candidate_facts":
                reusable,

            "revoked_reuse_facts":
                blocked
        }

    # =================================================
    # Session Snapshot
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

            "revoked_fact_ids":
                self.get_revoked_fact_ids(
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
