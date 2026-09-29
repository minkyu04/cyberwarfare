import json
from pathlib import Path

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


# =====================================================
# 기본 설정
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SAFE_RESPONSE = (
    "현재 접근 가능한 정보와 정보결합 정책을 기준으로 "
    "해당 내용을 제공할 수 없습니다."
)


def load_json(filename):
    with open(
        DATA_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# =====================================================
# Weighted Controlled RAG
# =====================================================

class WeightedControlledRAG:

    def __init__(self):

        # ---------------------------------------------
        # 1. 문서 및 Retrieval
        # ---------------------------------------------

        print("Loading permission-aware retriever...")

        self.retriever = PermissionAwareRetriever()

        self.documents = self.retriever.documents

        self.document_map = {
            doc["document_id"]: doc
            for doc in self.documents
        }

        # ---------------------------------------------
        # 2. 정보결합 통제 알고리즘
        # ---------------------------------------------

        print("Loading disclosure controller...")

        self.controller = (
            IncrementalDisclosureController()
        )

        # ---------------------------------------------
        # 3. Fact Catalog
        # ---------------------------------------------

        self.fact_catalog = (
            self.build_fact_catalog()
        )

        # ---------------------------------------------
        # 4. LLM
        # ---------------------------------------------

        print("Loading Qwen...")

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
        # 5. 세션 상태
        # ---------------------------------------------

        self.sessions = {}

        print("Weighted Controlled RAG ready.")

    # =================================================
    # Fact Catalog 생성
    # =================================================

    def build_fact_catalog(self):

        catalog = {}

        for doc in self.documents:

            for fact in doc.get(
                "protected_facts",
                []
            ):

                fact_id = fact["fact_id"]

                catalog[fact_id] = {
                    "fact_id": fact_id,
                    "fact": fact["fact"],
                    "category": fact["category"],
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

            self.sessions[session_id] = {
                "user_id": user_id,
                "exposed_facts": []
            }

        session = self.sessions[
            session_id
        ]

        # 동일 session_id를 다른 사용자가
        # 재사용하는 것을 차단한다.
        if session["user_id"] != user_id:

            raise ValueError(
                "Session user mismatch: "
                f"{session_id} is bound to "
                f"{session['user_id']}"
            )

        return session

    def reset_session(
        self,
        session_id
    ):

        if session_id in self.sessions:
            del self.sessions[session_id]

    # =================================================
    # 검색된 문서에서 Fact 후보 수집
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
            for result in retrieval_results
        }

        # ---------------------------------------------
        # 통제된 실험용 Ground Truth Mode
        # ---------------------------------------------
        #
        # 알고리즘 자체의 효과만 평가할 때는
        # 사람이 정의한 candidate fact를 입력할 수 있다.
        #
        # 단, 실제로 검색된 문서의 Fact만 허용한다.
        # ---------------------------------------------

        if forced_candidate_facts is not None:

            records = []

            for fact_id in forced_candidate_facts:

                if fact_id not in self.fact_catalog:

                    raise ValueError(
                        f"Unknown fact: {fact_id}"
                    )

                record = self.fact_catalog[
                    fact_id
                ]

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
                        "relevance_score": None
                    }
                )

            return records

        # ---------------------------------------------
        # End-to-End Mode
        # ---------------------------------------------
        #
        # 검색된 문서의 Fact들을 질문과 비교하여
        # 의미적으로 관련도가 높은 Fact 선택
        # ---------------------------------------------

        candidates = []

        for document_id in retrieved_ids:

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
                            doc["document_type"],

                        "title":
                            doc["title"]
                    }
                )

        if not candidates:
            return []

        # 질문 임베딩
        question_embedding = (
            self.retriever.model.encode(
                [question],
                normalize_embeddings=True
            )[0]
        )

        # Fact 임베딩
        fact_texts = [
            (
                f"문서 제목: {item['title']}\n"
                f"사실: {item['fact']}"
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

        for item, embedding in zip(
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
                    "relevance_score": score
                }
            )

        scored.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        return scored[
            :min(
                fact_top_k,
                len(scored)
            )
        ]

    # =================================================
    # 허용 Fact만으로 안전 Context 구성
    # =================================================

    def build_safe_context(
        self,
        retrieval_results,
        allowed_fact_ids
    ):

        allowed_fact_ids = set(
            allowed_fact_ids
        )

        sections = []

        for result in retrieval_results:

            document_id = (
                result["document_id"]
            )

            doc = self.document_map[
                document_id
            ]

            protected_facts = doc.get(
                "protected_facts",
                []
            )

            # -----------------------------------------
            # 보호 Fact가 없는 일반 공개문서
            # -----------------------------------------

            if not protected_facts:

                sections.append(
                    f"[문서 ID: {document_id}]\n"
                    f"제목: {doc['title']}\n"
                    f"내용: {doc['content']}"
                )

                continue

            # -----------------------------------------
            # 보호문서는 허용된 Fact만 전달
            # -----------------------------------------

            allowed_in_document = []

            for fact in protected_facts:

                if (
                    fact["fact_id"]
                    in allowed_fact_ids
                ):

                    allowed_in_document.append(
                        (
                            f"[{fact['fact_id']}] "
                            f"{fact['fact']}"
                        )
                    )

            if allowed_in_document:

                sections.append(
                    f"[문서 ID: {document_id}]\n"
                    f"제목: {doc['title']}\n"
                    "허용된 사실:\n"
                    + "\n".join(
                        allowed_in_document
                    )
                )

        return "\n\n".join(
            sections
        )

    # =================================================
    # Qwen 응답 생성
    # =================================================

    def generate_answer(
        self,
        question,
        safe_context
    ):

        if not safe_context.strip():

            return SAFE_RESPONSE

        system_prompt = """
당신은 합성 국방 업무지원 문서 질의응답 시스템이다.

반드시 [허용된 참고정보]에 포함된 사실만 사용하여
답변해야 한다.

참고정보에서 제외된 내용은 추측하거나 복원하거나
유추해서는 안 된다.

사용자가 요구했더라도 참고정보에 없는 값,
코드, 인물, 상태, 수치 또는 결론을 새로 생성하지 마라.

제공 가능한 사실만 자연스럽게 답변하라.

질문 전체에 답할 수 없는 경우에도 제공 가능한
사실만 답변하고, 나머지는 추측하지 마라.
"""

        user_prompt = f"""
[허용된 참고정보]

{safe_context}

[사용자 질문]

{question}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(
            self.model.device
        )

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **model_inputs,
                    max_new_tokens=180,
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
        # 1. Session 가져오기
        # ---------------------------------------------

        session = self.get_session(
            session_id=session_id,
            user_id=user_id
        )

        exposed_facts = list(
            session["exposed_facts"]
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
        # 3. Fact 후보 추출
        # ---------------------------------------------

        candidate_records = (
            self.collect_fact_candidates(
                question=question,
                retrieval_results=
                    retrieval_results,
                fact_top_k=fact_top_k,
                forced_candidate_facts=
                    forced_candidate_facts
            )
        )

        candidate_fact_ids = [
            item["fact_id"]
            for item in candidate_records
        ]

        # ---------------------------------------------
        # 4. 가중 최소손실 통제
        # ---------------------------------------------

        control_result = (
            self.controller.evaluate_turn(
                user_id=user_id,
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
        # 5. 허용 Fact만 Context로 구성
        # ---------------------------------------------

        safe_context = (
            self.build_safe_context(
                retrieval_results=
                    retrieval_results,
                allowed_fact_ids=
                    allowed_fact_ids
            )
        )

        # ---------------------------------------------
        # 6. 모든 보호 Fact가 제거된 경우
        # ---------------------------------------------

        if (
            candidate_fact_ids
            and not allowed_fact_ids
        ):

            response = SAFE_RESPONSE

        else:

            response = self.generate_answer(
                question=question,
                safe_context=safe_context
            )

        # ---------------------------------------------
        # 7. Session Ledger 갱신
        # ---------------------------------------------
        #
        # 현재 버전에서는 시스템이 실제 공개를
        # 승인한 Fact를 보수적으로 모두 기록한다.
        #
        # 향후 실제 자연어 출력에서 어떤 Fact가
        # 최종 전달되었는지 검증하는 Output Guard와
        # 연결할 예정이다.
        # ---------------------------------------------

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
        # 8. 최종 결과
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

            "response":
                response
        }


# =====================================================
# 결과 출력 함수
# =====================================================

def print_result(result):

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

    for item in result[
        "retrieved_documents"
    ]:

        print(
            f"- {item['document_id']} "
            f"(score={item['score']:.4f})"
        )

    print(
        "\nCandidate Facts:"
    )

    for item in result[
        "candidate_facts"
    ]:

        score = item[
            "relevance_score"
        ]

        if score is None:
            score_text = "GROUND_TRUTH"
        else:
            score_text = (
                f"{score:.4f}"
            )

        print(
            f"- {item['fact_id']} | "
            f"{item['fact']} | "
            f"relevance={score_text}"
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
        "\nSession Exposed Facts:",
        result[
            "session_exposed_facts"
        ]
    )

    print(
        "\nResponse:"
    )

    print(
        result["response"]
    )


# =====================================================
# 실행 Demonstration
# =====================================================

if __name__ == "__main__":

    rag = WeightedControlledRAG()

    # =================================================
    # TEST 1
    # 같은 Turn에서 가중 최소 업무손실 선택
    # =================================================

    print(
        "\n\n"
        "### TEST 1: "
        "Weighted Minimum-Loss Disclosure ###"
    )

    result = rag.answer(
        user_id="U3",
        session_id="WEIGHTED-DEMO",
        question=(
            "BLUE 훈련일과 통신장비 점검률, "
            "추가 확인 대상을 함께 알려줘."
        ),
        top_k=4,

        # 알고리즘 자체 검증을 위해
        # Ground Truth Fact를 사용
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
    # 다중턴 누적정보 통제
    # =================================================

    print(
        "\n\n"
        "### TEST 2: "
        "Multi-Turn Incremental Disclosure ###"
    )

    # Turn 1
    result_turn1 = rag.answer(
        user_id="U3",
        session_id="MULTITURN-DEMO",
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

    # Turn 2
    result_turn2 = rag.answer(
        user_id="U3",
        session_id="MULTITURN-DEMO",
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
    # 정상 권한 조합은 보존
    # =================================================

    print(
        "\n\n"
        "### TEST 3: "
        "Authorized Combination ###"
    )

    result = rag.answer(
        user_id="U4",
        session_id="AUTHORIZED-DEMO",
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
