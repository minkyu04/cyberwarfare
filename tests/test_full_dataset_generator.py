import unittest
from collections import Counter, defaultdict

from src.generate_full_dataset import (
    generate_users,
    generate_policies,
    generate_documents,
    generate_fact_utility,
    generate_inference_rules,
    generate_temporary_authorizations,
    generate_scenarios,
    validate_dataset
)


class TestFullDatasetGenerator(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.users = (
            generate_users()
        )

        cls.policies = (
            generate_policies()
        )

        cls.documents = (
            generate_documents()
        )

        cls.utilities = (
            generate_fact_utility(
                cls.documents
            )
        )

        cls.rules = (
            generate_inference_rules()
        )

        cls.temporary_authorizations = (
            generate_temporary_authorizations()
        )

        cls.scenarios = (
            generate_scenarios(
                cls.rules
            )
        )

    # =================================================
    # 1. User Count
    # =================================================

    def test_user_count(
        self
    ):

        self.assertEqual(
            len(
                self.users
            ),
            12
        )

    # =================================================
    # 2. Document Count
    # =================================================

    def test_document_count(
        self
    ):

        self.assertEqual(
            len(
                self.documents
            ),
            120
        )

    # =================================================
    # 3. 24 documents per type
    # =================================================

    def test_document_type_balance(
        self
    ):

        counts = Counter(
            doc[
                "document_type"
            ]
            for doc
            in self.documents
        )

        self.assertEqual(
            counts[
                "public_notice"
            ],
            24
        )

        self.assertEqual(
            counts[
                "personnel_document"
            ],
            24
        )

        self.assertEqual(
            counts[
                "operations_document"
            ],
            24
        )

        self.assertEqual(
            counts[
                "security_log"
            ],
            24
        )

        self.assertEqual(
            counts[
                "vulnerability_report"
            ],
            24
        )

    # =================================================
    # 4. Unique Document IDs
    # =================================================

    def test_document_ids_unique(
        self
    ):

        ids = [
            doc[
                "document_id"
            ]
            for doc
            in self.documents
        ]

        self.assertEqual(
            len(ids),
            len(
                set(
                    ids
                )
            )
        )

    # =================================================
    # 5. Unique Fact IDs
    # =================================================

    def test_fact_ids_unique(
        self
    ):

        fact_ids = [
            fact[
                "fact_id"
            ]
            for doc
            in self.documents
            for fact
            in doc.get(
                "protected_facts",
                []
            )
        ]

        self.assertEqual(
            len(
                fact_ids
            ),
            len(
                set(
                    fact_ids
                )
            )
        )

    # =================================================
    # 6. Utility for every protected Fact
    # =================================================

    def test_every_fact_has_utility(
        self
    ):

        fact_ids = {
            fact[
                "fact_id"
            ]
            for doc
            in self.documents
            for fact
            in doc.get(
                "protected_facts",
                []
            )
        }

        utility_ids = {
            item[
                "fact_id"
            ]
            for item
            in self.utilities
        }

        self.assertEqual(
            fact_ids,
            utility_ids
        )

    # =================================================
    # 7. Rule Count
    # =================================================

    def test_rule_count(
        self
    ):

        self.assertEqual(
            len(
                self.rules
            ),
            60
        )

    # =================================================
    # 8. Rule Facts Exist
    # =================================================

    def test_all_rule_facts_exist(
        self
    ):

        fact_ids = {
            fact[
                "fact_id"
            ]
            for doc
            in self.documents
            for fact
            in doc.get(
                "protected_facts",
                []
            )
        }

        for rule in (
            self.rules
        ):

            for fact_id in (
                rule[
                    "required_facts"
                ]
            ):

                self.assertIn(
                    fact_id,
                    fact_ids
                )

    # =================================================
    # 9. Overlapping Operations Rules Exist
    # =================================================

    def test_operations_rules_overlap(
        self
    ):

        operation_rules = [
            rule
            for rule in self.rules
            if (
                rule[
                    "domain"
                ]
                == "operations"
            )
        ]

        self.assertEqual(
            len(
                operation_rules
            ),
            24
        )

        # First two operations rules should share
        # the first schedule Fact.
        first_required = set(
            operation_rules[
                0
            ][
                "required_facts"
            ]
        )

        second_required = set(
            operation_rules[
                1
            ][
                "required_facts"
            ]
        )

        self.assertTrue(
            first_required
            & second_required
        )

    # =================================================
    # 10. Temporary Authorization Count
    # =================================================

    def test_temporary_authorization_count(
        self
    ):

        self.assertEqual(
            len(
                self.temporary_authorizations
            ),
            6
        )

    # =================================================
    # 11. Scenario Count
    # =================================================

    def test_scenario_count(
        self
    ):

        self.assertEqual(
            len(
                self.scenarios
            ),
            440
        )

    # =================================================
    # 12. Attack Distribution
    # =================================================

    def test_attack_distribution(
        self
    ):

        counts = Counter(
            scenario[
                "attack_type"
            ]
            for scenario
            in self.scenarios
        )

        self.assertEqual(
            counts[
                "none"
            ],
            120
        )

        for attack_type in [
            "AT1",
            "AT2",
            "AT3",
            "AT4",
            "AT5",
            "AT6",
            "AT7",
            "AT8"
        ]:

            self.assertEqual(
                counts[
                    attack_type
                ],
                40
            )

    # =================================================
    # 13. Dev/Test Split
    # =================================================

    def test_dev_test_split(
        self
    ):

        counts = Counter(
            scenario[
                "split"
            ]
            for scenario
            in self.scenarios
        )

        self.assertEqual(
            counts[
                "dev"
            ],
            88
        )

        self.assertEqual(
            counts[
                "test"
            ],
            352
        )

    # =================================================
    # 14. Multi-turn Session Split Integrity
    # =================================================

    def test_multiturn_sessions_not_split(
        self
    ):

        session_splits = defaultdict(
            set
        )

        for scenario in (
            self.scenarios
        ):

            session_splits[
                scenario[
                    "session_id"
                ]
            ].add(
                scenario[
                    "split"
                ]
            )

        for (
            session_id,
            splits
        ) in session_splits.items():

            self.assertEqual(
                len(
                    splits
                ),
                1,
                msg=(
                    f"Split leakage: "
                    f"{session_id}"
                )
            )

    # =================================================
    # 15. Full Validation
    # =================================================

    def test_full_validation(
        self
    ):

        summary = (
            validate_dataset(
                users=
                    self.users,

                policies=
                    self.policies,

                documents=
                    self.documents,

                rules=
                    self.rules,

                utilities=
                    self.utilities,

                temporary_authorizations=
                    self.temporary_authorizations,

                scenarios=
                    self.scenarios
            )
        )

        self.assertEqual(
            summary[
                "documents"
            ],
            120
        )

        self.assertEqual(
            summary[
                "scenarios"
            ],
            440
        )

        self.assertEqual(
            summary[
                "dev"
            ],
            88
        )

        self.assertEqual(
            summary[
                "test"
            ],
            352
        )


if __name__ == "__main__":

    unittest.main()
