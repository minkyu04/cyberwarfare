import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from src.permission_retrieval import (
    PermissionAwareRetriever
)

from src.incremental_disclosure import (
    IncrementalDisclosureController
)

from src.grounding_guard import (
    GroundedOutputGuard,
    SAFE_RESPONSE
)

from src.session_control import (
    RevocationAwareSessionManager
)


MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


class WeightedControlledRAG:

    def __init__(self):

        print(
            "Loading permission-aware retriever..."
        )

        self.retriever = (
            PermissionAwareRetriever()
        )

        self.documents = (
            self.retriever.documents
        )

        self.document_map = {
            doc["document_id"]:
                doc
            for doc
            in self.documents
        }

        print(
            "Loading disclosure controller..."
        )

        self.controller = (
            IncrementalDisclosureController()
        )

        print(
            "Loading revocation-aware session manager..."
        )

        self.session_manager = (
            RevocationAwareSessionManager(
                self.controller
            )
        )

        print(
            "Loading grounded output guard..."
        )

        self.output_guard = (
            GroundedOutputGuard()
        )

        self.fact_catalog = (
            self.build_fact_catalog()
        )

        print(
            "Loading Qwen..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto",
                device_map="auto"
            )
        )

        self.model.eval()

        torch.manual_seed(42)

        print(
            "Weighted Controlled RAG ready."
        )

    # =================================================
    # Fact Catalog
    # =================================================

    def build_fact_catalog(self):

        catalog = {}

        for doc in self.documents:

            for fact in doc.get(
                "protected_facts",
                []
            ):

                catalog[
                    fact["fact_id"]
                ] = {
                    "fact_id":
                        fact["fact_id"],

                    "fact":
                        fact["fact"],

                    "category":
                        fact["category"],

                    "sensitivity_weight":
                        fact["weight"],

                    "document_id":
                        doc["document_id"],

                    "document_type":
                        doc["document_type"],

                    "title":
                        doc["title"]
                }

        return catalog

    # =================================================
    # Session Reset
    # =================================================

    def reset_session(
        self,
        session_id
    ):

        self.session_manager.reset_session(
            session_id
        )

    # =================================================
    # Candidate Fact
    # =================================================

    def collect_fact_candidates(
        self,
        question,
        retrieval_results,
        fact_top_k=5,
        forced_candidate_facts=None
    ):

        retrieved_ids = {
            result["document_id"]
            for result
            in retrieval_results
        }

        # ---------------------------------------------
        # Ground Truth Mode
        # ---------------------------------------------

        if (
            forced_candidate_facts
            is not None
        ):

            records = []

            for fact_id in (
                forced_candidate_facts
            ):

                if (
                    fact_id
                    not in self.fact_catalog
                ):

                    raise ValueError(
                        f"Unknown fact: "
                        f"{fact_id}"
                    )

                record = (
                    self.fact_catalog[
                        fact_id
                    ]
                )

                if (
                    record["document_id"]
                    not in retrieved_ids
                ):

                    raise ValueError(
                        f"{fact_id} belongs to "
                        f"{record['document_id']}, "
                        "but that document was "
                        "not retrieved."
                    )

                records.append(
                    {
                        **record,
                        "relevance_score":
                            None
                    }
                )

            return records

        # ---------------------------------------------
        # End-to-End Mode
        # ---------------------------------------------

        candidates = []

        for document_id in (
            retrieved_ids
        ):

            doc = (
                self.document_map[
                    document_id
                ]
            )

            for fact in doc.get(
                "protected_facts",
                []
            ):

                candidates.append(
                    {
                        "fact_id":
                            fact["fact_id"],

                        "fact":
                            fact["fact"],

                        "category":
                            fact["category"],

                        "sensitivity_weight":
                            fact["weight"],

                        "document_id":
                            document_id,

                        "document_type":
                            doc[
                                "document_type"
                            ],

                        "title":
                            doc["title"]
                    }
                )

        if not candidates:

            return []

        question_embedding = (
            self.retriever.model.encode(
                [question],
                normalize_embeddings=True
            )[0]
        )

        fact_texts = [
            (
                f"문서 제목: "
                f"{item['title']}\n"
                f"사실: "
                f"{item['fact']}"
            )
            for item
            in candidates
        ]

        fact_embeddings = (
            self.retriever.model.encode(
                fact_texts,
                normalize_embeddings=True
            )
        )

        scored = []

        for (
            item,
            embedding
        ) in zip(
            candidates,
            fact_embeddings
        ):

            score = float(
                np.dot(
                    question_embedding,
                    embedding
                )
            )

            scored.append(
                {
                    **item,
                    "relevance_score":
                        score
                }
            )

        scored.sort(
            key=lambda x:
                x[
                    "relevance_score"
                ],
            reverse=True
        )

        return scored[
            :min(
                fact_top_k,
                len(scored)
            )
        ]

    # =================================================
    # Fact Context
    # =================================================

    def build_fact_context(
        self,
        allowed_fact_ids
    ):

        if not allowed_fact_ids:

            return ""

        document_groups = {}

        for fact_id in (
            allowed_fact_ids
        ):

            record = (
                self.fact_catalog[
                    fact_id
                ]
            )

            document_id = (
                record[
                    "document_id"
                ]
            )

            if (
                document_id
                not in document_groups
            ):

                document_groups[
                    document_id
                ] = {
                    "title":
                        record[
                            "title"
                        ],

                    "facts":
                        []
                }

            document_groups[
                document_id
            ]["facts"].append(
                (
                    f"[{fact_id}] "
                    f"{record['fact']}"
                )
            )

        sections = []

        for (
            document_id,
            data
        ) in document_groups.items():

            sections.append(
                f"[문서 ID: "
                f"{document_id}]\n"
                f"제목: "
                f"{data['title']}\n"
                "허용된 사실:\n"
                + "\n".join(
                    data[
                        "facts"
                    ]
                )
            )

        return "\n\n".join(
            sections
        )

    # =================================================
    # Public Context
    # =================================================

    def build_public_context(
        self,
        retrieval_results,
        max_documents=2
    ):

        sections = []

        for result in (
            retrieval_results
        ):

            doc = (
                self.document_map[
                    result[
                        "document_id"
                    ]
                ]
            )

            if doc.get(
                "protected_facts",
                []
            ):

                continue

            sections.append(
                f"[문서 ID: "
                f"{doc['document_id']}]\n"
                f"제목: "
                f"{doc['title']}\n"
                f"내용: "
                f"{doc['content']}"
            )

            if (
                len(sections)
                >= max_documents
            ):

                break

        return "\n\n".join(
            sections
        )

    # =================================================
    # Public Fallback
    # =================================================

    def build_public_fallback(
        self,
        retrieval_results,
        max_documents=2
    ):

        texts = []

        for result in (
            retrieval_results
        ):

            doc = (
                self.document_map[
                    result[
                        "document_id"
                    ]
                ]
            )

            if doc.get(
                "protected_facts",
                []
            ):

                continue

            texts.append(
                doc[
                    "content"
                ]
            )

            if (
                len(texts)
                >= max_documents
            ):

                break

        if not texts:

            return SAFE_RESPONSE

        return (
            "확인 가능한 공개정보는 다음과 같습니다. "
            + " ".join(
                texts
            )
        )

    # =================================================
    # Safe Context
    # =================================================

    def build_safe_context(
        self,
        retrieval_results,
        candidate_fact_ids,
        allowed_fact_ids
    ):

        if candidate_fact_ids:

            return (
                self.build_fact_context(
                    allowed_fact_ids
                )
            )

        return (
            self.build_public_context(
                retrieval_results
            )
        )

    # =================================================
    # Qwen Draft
    # =================================================

    def generate_draft(
        self,
        question,
        safe_context
    ):

        if not safe_context.strip():

            return SAFE_RESPONSE

        system_prompt = """
당신은 합성 국방 업무지원 질의응답 시스템이다.

반드시 [허용된 참고정보]에 명시된 사실만 사용하여
질문에 답변하라.

현재 허용된 참고정보에 없는 수치, 코드, 날짜,
시간, 비율, 인물, 장비, 상태 또는 결론을
새로 생성하지 마라.

과거 대화에서 제공된 정보라도 현재 참고정보에
없다면 다시 출력하지 마라.

허용된 사실만 간단하고 직접적으로 전달하라.
"""

        user_prompt = f"""
[허용된 참고정보]

{safe_context}

[질문]

{question}

현재 허용된 참고정보에 존재하는 사실만 이용하여
답변하라.
"""

        messages = [
            {
                "role":
                    "system",

                "content":
                    system_prompt
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt
            }
        ]

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        model_inputs = (
            self.tokenizer(
                [text],
                return_tensors="pt"
            ).to(
                self.model.device
            )
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=
                        self.tokenizer
                        .eos_token_id
                )
            )

        generated_ids = [
            output_ids[
                len(input_ids):
            ]

            for (
                input_ids,
                output_ids
            )
            in zip(
                model_inputs.input_ids,
                generated_ids
            )
        ]

        response = (
            self.tokenizer
            .batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        return response.strip()

    # =================================================
    # Answer
    # =================================================

    def answer(
        self,
        user_id,
        session_id,
        question,
        top_k=4,
        fact_top_k=5,
        forced_candidate_facts=None,
        as_of=None
    ):

        # ---------------------------------------------
        # Session
        # ---------------------------------------------

        self.session_manager.get_session(
            session_id,
            user_id
        )

        exposed_facts_before = (
            self.session_manager
            .get_exposed_fact_ids(
                session_id=
                    session_id,

                user_id=
                    user_id
            )
        )

        authorization_context = (
            self.controller
            .get_authorization_context(
                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        session_before = (
            self.session_manager
            .get_session_snapshot(
                session_id=
                    session_id,

                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        # ---------------------------------------------
        # Retrieval
        # ---------------------------------------------

        retrieval_results = (
            self.retriever.search(
                user_id=
                    user_id,

                query=
                    question,

                top_k=
                    top_k
            )
        )

        # ---------------------------------------------
        # Candidate Fact
        # ---------------------------------------------

        candidate_records = (
            self.collect_fact_candidates(
                question=
                    question,

                retrieval_results=
                    retrieval_results,

                fact_top_k=
                    fact_top_k,

                forced_candidate_facts=
                    forced_candidate_facts
            )
        )

        candidate_fact_ids = [
            item["fact_id"]
            for item
            in candidate_records
        ]

        # ---------------------------------------------
        # Information Combination Control
        # ---------------------------------------------

        control_result = (
            self.controller
            .evaluate_turn(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts_before,

                candidate_facts=
                    candidate_fact_ids,

                as_of=
                    as_of
            )
        )

        removed_fact_ids = (
            control_result[
                "removed_facts"
            ]
        )

        allowed_fact_ids = (
            control_result[
                "allowed_facts"
            ]
        )

        # ---------------------------------------------
        # Context
        # ---------------------------------------------

        safe_context = (
            self.build_safe_context(
                retrieval_results=
                    retrieval_results,

                candidate_fact_ids=
                    candidate_fact_ids,

                allowed_fact_ids=
                    allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # Draft
        # ---------------------------------------------

        if (
            candidate_fact_ids
            and not allowed_fact_ids
        ):

            draft_response = (
                SAFE_RESPONSE
            )

        else:

            draft_response = (
                self.generate_draft(
                    question=
                        question,

                    safe_context=
                        safe_context
                )
            )

        # ---------------------------------------------
        # Fallback 결정
        # ---------------------------------------------

        if candidate_fact_ids:

            fallback_response = None

        else:

            fallback_response = (
                self.build_public_fallback(
                    retrieval_results
                )
            )

        # ---------------------------------------------
        # Grounded Output Guard
        #
        # 보호 Fact 여부와 관계없이 항상 적용
        # ---------------------------------------------

        guard_result = (
            self.output_guard
            .validate(
                question=
                    question,

                allowed_fact_ids=
                    allowed_fact_ids,

                removed_fact_ids=
                    removed_fact_ids,

                draft_response=
                    draft_response,

                additional_allowed_text=
                    safe_context,

                fallback_response=
                    fallback_response
            )
        )

        final_response = (
            guard_result[
                "final_response"
            ]
        )

        # ---------------------------------------------
        # Exposure 기록
        # ---------------------------------------------

        self.session_manager.record_disclosure(
            session_id=
                session_id,

            user_id=
                user_id,

            disclosed_fact_ids=
                allowed_fact_ids,

            exposed_facts_before=
                exposed_facts_before,

            candidate_facts=
                candidate_fact_ids,

            as_of=
                as_of
        )

        session_after = (
            self.session_manager
            .get_session_snapshot(
                session_id=
                    session_id,

                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        # ---------------------------------------------
        # 결과
        # ---------------------------------------------

        return {
            "user_id":
                user_id,

            "session_id":
                session_id,

            "as_of":
                (
                    str(as_of)
                    if as_of is not None
                    else None
                ),

            "authorization_context":
                authorization_context,

            "question":
                question,

            "retrieved_documents":
                [
                    {
                        "document_id":
                            result[
                                "document_id"
                            ],

                        "score":
                            result[
                                "score"
                            ]
                    }

                    for result
                    in retrieval_results
                ],

            "candidate_facts":
                candidate_records,

            "previously_exposed_facts":
                exposed_facts_before,

            "temporary_rule_ids":
                session_after[
                    "temporary_rule_ids"
                ],

            "revoked_rule_ids":
                session_before[
                    "revoked_rule_ids"
                ],

            "detected_rules":
                control_result[
                    "detected_rules"
                ],

            "preexisting_violations":
                control_result[
                    "preexisting_violations"
                ],

            "current_response_violations":
                control_result[
                    "current_response_violations"
                ],

            "removed_facts":
                removed_fact_ids,

            "allowed_facts":
                allowed_fact_ids,

            "candidate_utility":
                control_result[
                    "candidate_utility"
                ],

            "removed_utility":
                control_result[
                    "removed_utility"
                ],

            "retained_utility":
                control_result[
                    "retained_utility"
                ],

            "utility_retention_rate":
                control_result[
                    "utility_retention_rate"
                ],

            "guard":
                guard_result,

            "safe_context":
                safe_context,

            "draft_response":
                draft_response,

            "response":
                final_response,

            "session_exposed_facts":
                session_after[
                    "exposed_fact_ids"
                ],

            "session_events":
                session_after[
                    "exposure_events"
                ]
        }


# =====================================================
# 출력
# =====================================================

def print_result(
    result
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "USER:",
        result["user_id"]
    )

    print(
        "SESSION:",
        result["session_id"]
    )

    print(
        "TIME:",
        result["as_of"]
    )

    print(
        "\nAuthorization:"
    )

    print(
        result[
            "authorization_context"
        ]
    )

    print(
        "\nQuestion:"
    )

    print(
        result[
            "question"
        ]
    )

    print(
        "\nPrevious Exposure:"
    )

    print(
        result[
            "previously_exposed_facts"
        ]
    )

    print(
        "Temporary Rules:"
    )

    print(
        result[
            "temporary_rule_ids"
        ]
    )

    print(
        "Revoked Rules:"
    )

    print(
        result[
            "revoked_rule_ids"
        ]
    )

    print(
        "\nDetected Rules:"
    )

    print(
        result[
            "detected_rules"
        ]
    )

    print(
        "Preexisting Violations:"
    )

    print(
        result[
            "preexisting_violations"
        ]
    )

    print(
        "Current Response Violations:"
    )

    print(
        result[
            "current_response_violations"
        ]
    )

    print(
        "\nRemoved:"
    )

    print(
        result[
            "removed_facts"
        ]
    )

    print(
        "Allowed:"
    )

    print(
        result[
            "allowed_facts"
        ]
    )

    print(
        "\nUtility:"
    )

    print(
        "Candidate:",
        result[
            "candidate_utility"
        ]
    )

    print(
        "Removed:",
        result[
            "removed_utility"
        ]
    )

    print(
        "Retained:",
        result[
            "retained_utility"
        ]
    )

    print(
        "Retention Rate:",
        f"{result['utility_retention_rate']:.4f}"
    )

    print(
        "\nGuard:"
    )

    print(
        "Triggered:",
        result[
            "guard"
        ][
            "guard_triggered"
        ]
    )

    print(
        "Mode:",
        result[
            "guard"
        ][
            "response_mode"
        ]
    )

    print(
        "Unsupported:",
        result[
            "guard"
        ][
            "unsupported_tokens"
        ]
    )

    print(
        "\nDraft:"
    )

    print(
        result[
            "draft_response"
        ]
    )

    print(
        "\nFinal Response:"
    )

    print(
        result[
            "response"
        ]
    )

    print(
        "\nExposure History:"
    )

    print(
        result[
            "session_exposed_facts"
        ]
    )


# =====================================================
# Demonstration
# =====================================================

if __name__ == "__main__":

    rag = WeightedControlledRAG()

    # =================================================
    # TURN 1
    # 임시 권한 활성
    # =================================================

    result1 = rag.answer(
        user_id=
            "U3",

        session_id=
            "REVOCATION-DEMO",

        question=(
            "BLUE 훈련일과 "
            "통신장비 점검률을 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2",
            "OPS002-F2"
        ],

        as_of=(
            "2026-09-29T10:00:00"
            "+09:00"
        )
    )

    print_result(
        result1
    )

    # =================================================
    # TURN 2
    # 권한 활성 유지
    # =================================================

    result2 = rag.answer(
        user_id=
            "U3",

        session_id=
            "REVOCATION-DEMO",

        question=(
            "방금 훈련일과 "
            "점검률을 다시 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2",
            "OPS002-F2"
        ],

        as_of=(
            "2026-09-29T11:00:00"
            "+09:00"
        )
    )

    print_result(
        result2
    )

    # =================================================
    # TURN 3
    # 임시권한 만료
    # =================================================

    result3 = rag.answer(
        user_id=
            "U3",

        session_id=
            "REVOCATION-DEMO",

        question=(
            "방금 훈련일과 "
            "점검률을 다시 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2",
            "OPS002-F2"
        ],

        as_of=(
            "2026-09-29T12:30:00"
            "+09:00"
        )
    )

    print_result(
        result3
    )

    # =================================================
    # TURN 4
    # 안전한 단일 Fact
    # =================================================

    result4 = rag.answer(
        user_id=
            "U3",

        session_id=
            "REVOCATION-DEMO",

        question=(
            "BLUE 훈련일만 다시 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2"
        ],

        as_of=(
            "2026-09-29T12:40:00"
            "+09:00"
        )
    )

    print_result(
        result4
    )
