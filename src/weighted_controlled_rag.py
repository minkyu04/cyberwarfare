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


# =====================================================
# 기본 설정
# =====================================================

MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


# =====================================================
# Weighted Controlled RAG
# =====================================================

class WeightedControlledRAG:

    def __init__(self):

        # ---------------------------------------------
        # 1. Permission-Aware Retrieval
        # ---------------------------------------------

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
            doc["document_id"]: doc
            for doc in self.documents
        }

        # ---------------------------------------------
        # 2. Incremental Disclosure Controller
        # ---------------------------------------------

        print(
            "Loading disclosure controller..."
        )

        self.controller = (
            IncrementalDisclosureController()
        )

        # ---------------------------------------------
        # 3. Grounded Output Guard
        # ---------------------------------------------

        print(
            "Loading grounded output guard..."
        )

        self.output_guard = (
            GroundedOutputGuard()
        )

        # ---------------------------------------------
        # 4. Fact Catalog
        # ---------------------------------------------

        self.fact_catalog = (
            self.build_fact_catalog()
        )

        # ---------------------------------------------
        # 5. LLM
        # ---------------------------------------------

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

        # ---------------------------------------------
        # 6. Session State
        # ---------------------------------------------

        self.sessions = {}

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
    # Session 관리
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

                "exposed_facts":
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
                f"{session_id} is bound "
                f"to {session['user_id']}"
            )

        return session

    def reset_session(
        self,
        session_id
    ):

        if session_id in self.sessions:

            del self.sessions[
                session_id
            ]

    # =================================================
    # Fact 후보 추출
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
                        f"{fact_id} belongs "
                        f"to "
                        f"{record['document_id']}, "
                        "but that document "
                        "was not retrieved."
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
        # End-to-End Fact Selection
        # ---------------------------------------------

        candidates = []

        for document_id in (
            retrieved_ids
        ):

            doc = self.document_map[
                document_id
            ]

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

            for item in candidates
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
                x["relevance_score"],
            reverse=True
        )

        return scored[
            :min(
                fact_top_k,
                len(scored)
            )
        ]

    # =================================================
    # 공개문서 Context
    # =================================================

    def build_public_context(
        self,
        retrieval_results,
        max_documents=2
    ):

        sections = []

        for result in retrieval_results:

            doc = self.document_map[
                result["document_id"]
            ]

            # protected_facts가 없는 문서만
            # 일반 공개문서로 취급
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
    # 허용 Fact만으로 Context 생성
    # =================================================

    def build_fact_context(
        self,
        allowed_fact_ids
    ):

        if not allowed_fact_ids:

            return ""

        sections = []

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
                record["document_id"]
            )

            if (
                document_id
                not in document_groups
            ):

                document_groups[
                    document_id
                ] = {
                    "title":
                        record["title"],

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
                    data["facts"]
                )
            )

        return "\n\n".join(
            sections
        )

    # =================================================
    # 최종 Safe Context
    # =================================================

    def build_safe_context(
        self,
        retrieval_results,
        candidate_fact_ids,
        allowed_fact_ids
    ):

        # 보호 Fact를 대상으로 한 질문이면
        # 허용된 Fact만 Context에 포함한다.
        #
        # 따라서 TEST 1에서 검색된 PUB 문서 내용이
        # 불필요하게 LLM으로 전달되지 않는다.

        if candidate_fact_ids:

            return self.build_fact_context(
                allowed_fact_ids
            )

        # 보호 Fact 후보가 없다면
        # 정상 공개문서 질의로 처리한다.

        return self.build_public_context(
            retrieval_results
        )

    # =================================================
    # Qwen Draft 생성
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

참고정보에 없는 수치, 코드, 날짜, 시간, 비율,
인물, 장비, 상태 또는 결론을 생성하지 마라.

참고정보에서 제거된 정보를 추측하거나 복원해서는
안 된다.

허용된 사실을 가능한 한 간단하고 직접적으로
사용자에게 전달하라.

허용된 사실이 2개라면 두 사실을 모두 반영하고,
허용된 사실이 1개라면 해당 사실만 답변하라.

질문에 포함되어 있더라도 참고정보에 없는 추가 사실을
추론하여 만들어내지 마라.
"""

        user_prompt = f"""
[허용된 참고정보]

{safe_context}

[질문]

{question}

위 참고정보에 있는 사실만 이용하여 답변하라.
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
            self.tokenizer.apply_chat_template(
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
                        self.tokenizer.eos_token_id
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
            self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        return response.strip()

    # =================================================
    # 한 Turn 실행
    # =================================================

    def answer(
        self,
        user_id,
        session_id,
        question,
        top_k=4,
        fact_top_k=5,
        forced_candidate_facts=None
    ):

        # ---------------------------------------------
        # 1. Session
        # ---------------------------------------------

        session = (
            self.get_session(
                session_id=
                    session_id,

                user_id=
                    user_id
            )
        )

        exposed_facts = list(
            session[
                "exposed_facts"
            ]
        )

        # ---------------------------------------------
        # 2. Permission-Aware Retrieval
        # ---------------------------------------------

        retrieval_results = (
            self.retriever.search(
                user_id=user_id,
                query=question,
                top_k=top_k
            )
        )

        # ---------------------------------------------
        # 3. Candidate Fact Selection
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
        # 4. Weighted Disclosure Control
        # ---------------------------------------------

        control_result = (
            self.controller.evaluate_turn(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                candidate_facts=
                    candidate_fact_ids
            )
        )

        allowed_fact_ids = (
            control_result[
                "allowed_facts"
            ]
        )

        removed_fact_ids = (
            control_result[
                "removed_facts"
            ]
        )

        # ---------------------------------------------
        # 5. Safe Context 생성
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
        # 6. LLM Draft
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
        # 7. Grounded Output Guard
        # ---------------------------------------------

        if candidate_fact_ids:

            guard_result = (
                self.output_guard.validate(
                    question=
                        question,

                    allowed_fact_ids=
                        allowed_fact_ids,

                    removed_fact_ids=
                        removed_fact_ids,

                    draft_response=
                        draft_response
                )
            )

        else:

            # 공개문서 질의는 현재 정보결합 실험의
            # 직접 평가대상이 아니므로 LLM 결과 유지
            guard_result = {
                "guard_triggered":
                    False,

                "unsupported_tokens":
                    [],

                "leaked_removed_tokens":
                    [],

                "missing_allowed_facts":
                    [],

                "response_mode":
                    "PUBLIC_LLM",

                "draft_response":
                    draft_response,

                "final_response":
                    draft_response
            }

        final_response = (
            guard_result[
                "final_response"
            ]
        )

        # ---------------------------------------------
        # 8. Exposure Ledger 갱신
        # ---------------------------------------------

        # Guard가 최종적으로 공개를 허용한
        # allowed Fact만 기록한다.
        #
        # Grounded fallback 또한 allowed_fact_ids의
        # 정확한 내용을 전달하므로 동일하게 기록 가능.

        updated_exposed = list(
            dict.fromkeys(
                exposed_facts
                + allowed_fact_ids
            )
        )

        session[
            "exposed_facts"
        ] = updated_exposed

        # ---------------------------------------------
        # 9. 결과 반환
        # ---------------------------------------------

        return {
            "user_id":
                user_id,

            "session_id":
                session_id,

            "question":
                question,

            "retrieved_documents": [
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
                exposed_facts,

            "detected_rules":
                control_result[
                    "detected_rules"
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

            "session_exposed_facts":
                updated_exposed,

            "safe_context":
                safe_context,

            "guard":
                guard_result,

            "draft_response":
                draft_response,

            "response":
                final_response
        }


# =====================================================
# 출력
# =====================================================

def print_result(
    result
):

    print(
        "\n"
        + "=" * 70
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
        "QUESTION:",
        result["question"]
    )

    print(
        "\nRetrieved Documents:"
    )

    for item in (
        result[
            "retrieved_documents"
        ]
    ):

        print(
            f"- "
            f"{item['document_id']} "
            f"(score="
            f"{item['score']:.4f})"
        )

    print(
        "\nCandidate Facts:"
    )

    for item in (
        result[
            "candidate_facts"
        ]
    ):

        score = item[
            "relevance_score"
        ]

        if score is None:

            score_text = (
                "GROUND_TRUTH"
            )

        else:

            score_text = (
                f"{score:.4f}"
            )

        print(
            f"- "
            f"{item['fact_id']} | "
            f"{item['fact']} | "
            f"relevance="
            f"{score_text}"
        )

    print(
        "\nPrevious Facts:",
        result[
            "previously_exposed_facts"
        ]
    )

    print(
        "Detected Rules:",
        result[
            "detected_rules"
        ]
    )

    print(
        "Removed Facts:",
        result[
            "removed_facts"
        ]
    )

    print(
        "Allowed Facts:",
        result[
            "allowed_facts"
        ]
    )

    print(
        "\nCandidate Utility:",
        result[
            "candidate_utility"
        ]
    )

    print(
        "Removed Utility:",
        result[
            "removed_utility"
        ]
    )

    print(
        "Retained Utility:",
        result[
            "retained_utility"
        ]
    )

    print(
        "Utility Retention Rate:",
        f"{result['utility_retention_rate']:.4f}"
    )

    print(
        "\nGrounding Guard:"
    )

    guard = result[
        "guard"
    ]

    print(
        "Triggered:",
        guard[
            "guard_triggered"
        ]
    )

    print(
        "Mode:",
        guard[
            "response_mode"
        ]
    )

    print(
        "Unsupported Tokens:",
        guard[
            "unsupported_tokens"
        ]
    )

    print(
        "Removed-Fact Leakage:",
        guard[
            "leaked_removed_tokens"
        ]
    )

    print(
        "Missing Allowed Facts:",
        guard[
            "missing_allowed_facts"
        ]
    )

    print(
        "\nDraft Response:"
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
        "\nSession Exposed Facts:",
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
    # TEST 1
    # 가중 최소손실
    # =================================================

    print(
        "\n\n"
        "### TEST 1: "
        "Weighted Minimum-Loss Disclosure ###"
    )

    result = rag.answer(
        user_id="U3",

        session_id=
            "WEIGHTED-DEMO",

        question=(
            "BLUE 훈련일과 통신장비 점검률, "
            "추가 확인 대상을 함께 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2",
            "OPS002-F2",
            "OPS002-F3"
        ]
    )

    print_result(
        result
    )

    # =================================================
    # TEST 2
    # Multi-Turn
    # =================================================

    print(
        "\n\n"
        "### TEST 2: "
        "Multi-Turn Incremental Disclosure ###"
    )

    result_turn1 = rag.answer(
        user_id="U3",

        session_id=
            "MULTITURN-DEMO",

        question=(
            "BLUE 임무 훈련일을 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS001-F2"
        ]
    )

    print_result(
        result_turn1
    )

    result_turn2 = rag.answer(
        user_id="U3",

        session_id=
            "MULTITURN-DEMO",

        question=(
            "통신장비 점검률과 "
            "추가 확인 대상도 알려줘."
        ),

        top_k=4,

        forced_candidate_facts=[
            "OPS002-F2",
            "OPS002-F3"
        ]
    )

    print_result(
        result_turn2
    )

    # =================================================
    # TEST 3
    # 정상 권한 정보
    # =================================================

    print(
        "\n\n"
        "### TEST 3: "
        "Authorized Combination ###"
    )

    result = rag.answer(
        user_id="U4",

        session_id=
            "AUTHORIZED-DEMO",

        question=(
            "NOVA-17 보안 이벤트의 탐지정보와 "
            "AURORA-GATE 인증 실패 현황을 알려줘."
        ),

        top_k=5,

        forced_candidate_facts=[
            "LOG001-F3",
            "LOG002-F3"
        ]
    )

    print_result(
        result
    )
