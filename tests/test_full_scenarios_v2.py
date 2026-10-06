import unittest
from collections import Counter

from src.generate_full_scenarios_v2 import (
    DATA_DIR,
    DOCUMENT_FILE,
    RULE_FILE,
    load_json,
    generate_all_scenarios,
    validate_scenarios
)


class TestFullScenariosV2(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls
    ):

        cls.documents = (
            load_json(
                DOCUMENT_FILE
            )
        )

        cls.rules = (
            load_json(
                RULE_FILE
            )
        )

        cls.scenarios = (
            generate_all_scenarios(
                cls.rules
            )
        )

    def test_total_rows(
        self
    ):

        self.assertEqual(
            len(
                self.scenarios
            ),
            440
        )

    def test_split(
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

    def test_attack_counts(
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

    def test_target_documents_exist(
        self
    ):

        document_ids = {
            doc[
                "document_id"
            ]
            for doc
            in self.documents
        }

        for scenario in (
            self.scenarios
        ):

            for document_id in (
                scenario[
                    "target_document_ids"
                ]
            ):

                self.assertIn(
                    document_id,
                    document_ids
                )

    def test_at7_has_document_anchors(
        self
    ):

        at7 = [
            scenario
            for scenario
            in self.scenarios
            if (
                scenario[
                    "attack_type"
                ]
                == "AT7"
            )
        ]

        for scenario in at7:

            for document_id in (
                scenario[
                    "target_document_ids"
                ]
            ):

                self.assertIn(
                    document_id,
                    scenario[
                        "question"
                    ]
                )

    def test_at8_has_document_anchors(
        self
    ):

        at8 = [
            scenario
            for scenario
            in self.scenarios
            if (
                scenario[
                    "attack_type"
                ]
                == "AT8"
            )
        ]

        for scenario in at8:

            for document_id in (
                scenario[
                    "target_document_ids"
                ]
            ):

                self.assertIn(
                    document_id,
                    scenario[
                        "question"
                    ]
                )

    def test_query_variants_change_text(
        self
    ):

        variants = {
            scenario[
                "query_variant"
            ]
            for scenario
            in self.scenarios
        }

        self.assertIn(
            "normal",
            variants
        )

        self.assertIn(
            "typo",
            variants
        )

        self.assertIn(
            "abbreviation",
            variants
        )

        self.assertIn(
            "ambiguous",
            variants
        )

    def test_full_validation(
        self
    ):

        result = (
            validate_scenarios(
                scenarios=
                    self.scenarios,

                documents=
                    self.documents,

                rules=
                    self.rules
            )
        )

        self.assertEqual(
            result[
                "rows"
            ],
            440
        )

        self.assertEqual(
            result[
                "dev"
            ],
            88
        )

        self.assertEqual(
            result[
                "test"
            ],
            352
        )

        self.assertEqual(
            result[
                "at7_domain_counts"
            ][
                "personnel"
            ],
            10
        )

        self.assertEqual(
            result[
                "at7_domain_counts"
            ][
                "operations"
            ],
            10
        )

        self.assertEqual(
            result[
                "at7_domain_counts"
            ][
                "security_log"
            ],
            10
        )

        self.assertEqual(
            result[
                "at7_domain_counts"
            ][
                "vulnerability"
            ],
            10
        )

        self.assertEqual(
            result[
                "at8_cumulative_domain_counts"
            ][
                "personnel"
            ],
            6
        )

        self.assertEqual(
            result[
                "at8_cumulative_domain_counts"
            ][
                "operations"
            ],
            6
        )

        self.assertEqual(
            result[
                "at8_cumulative_domain_counts"
            ][
                "security_log"
            ],
            6
        )

        self.assertEqual(
            result[
                "at8_cumulative_domain_counts"
            ][
                "vulnerability"
            ],
            6
        )


if __name__ == "__main__":

    unittest.main()
