import json
import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from database import (
    authenticate_customer,
    get_full_customer_data,
    build_full_customer_context,
    insert_audit_log,
)

load_dotenv()

APP_NAME = "SecureLife Assist 🛡️"
INDEX_DIR = "insurance_faiss_index"
CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CUSTOMER_SUPPORT_PHONE = "19000"

NOT_FOUND_MESSAGE = (
    "I could not find this information in the available SecureLife records. "
    f"Please contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE} for further assistance."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "customer_contract" not in st.session_state:
    st.session_state.customer_contract = None

if "verification" not in st.session_state:
    st.session_state.verification = {
        "stage": None,
        "policy_number": None,
        "question": None,
        "question_type": None,
    }

if "example_question" not in st.session_state:
    st.session_state.example_question = None


def api_key_exists() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY"))


def knowledge_base_exists() -> bool:
    return os.path.isdir(INDEX_DIR) and len(os.listdir(INDEX_DIR)) > 0


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm(temperature: float = 0.2, max_tokens: int = 700):
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def message_to_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content)


def load_vectorstore():
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve_rag_context(question: str, k: int = 3) -> str:
    if not knowledge_base_exists():
        return ""

    vectorstore = load_vectorstore()
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def get_spinner_text(question_type: str) -> str:
    if question_type == "general":
        return "Reviewing SecureLife policy documents..."
    if question_type == "personal":
        return "Checking your SecureLife policy records..."
    if question_type == "both":
        return "Checking your policy records and reviewing SecureLife documents..."
    return "Preparing your answer..."


def is_quota_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return (
        "resource_exhausted" in error_text
        or "429" in error_text
        or "quota" in error_text
    )


def is_not_found_answer(answer: str) -> bool:
    text = answer.lower()
    phrases = [
        "i could not find",
        "could not find this information",
        "not found",
        "not available",
        "contact securelife customer support",
        "contact customer support",
    ]
    return any(phrase in text for phrase in phrases)


def reset_verification() -> None:
    st.session_state.verification = {
        "stage": None,
        "policy_number": None,
        "question": None,
        "question_type": None,
    }

def looks_like_policy_number(user_message: str) -> bool:
    text = user_message.strip().upper()

    return bool(re.fullmatch(r"SL-[A-Z]+-\d+", text))


def looks_like_access_code(user_message: str) -> bool:
    text = user_message.strip()

    return bool(re.fullmatch(r"\d{4,8}", text))


def cancel_verification_and_continue() -> None:
    reset_verification()


def start_verification(question: str, question_type: str) -> None:
    st.session_state.verification = {
        "stage": "waiting_policy_number",
        "policy_number": None,
        "question": question,
        "question_type": question_type,
    }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "To answer this securely, I need to verify your policy first. "
                "Please enter your policy number."
            ),
        }
    )

    st.rerun()


def handle_verification_message(user_message: str) -> None:
    verification = st.session_state.verification

    if verification["stage"] == "waiting_policy_number":
        policy_number = user_message.strip().upper()

        verification["policy_number"] = policy_number
        verification["stage"] = "waiting_access_code"

        st.session_state.messages.append(
            {"role": "user", "content": f"Policy number: {policy_number}"}
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": "Thank you. Please enter your access code."}
        )

        st.rerun()

    elif verification["stage"] == "waiting_access_code":
        access_code = user_message.strip()
        policy_number = verification["policy_number"]

        st.session_state.messages.append(
            {"role": "user", "content": "Access code: ••••"}
        )

        try:
            contract = authenticate_customer(
                policy_number=policy_number,
                access_code=access_code,
            )
        except Exception as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I could not verify the policy right now. "
                        f"Please contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}. "
                        f"Details: {exc}"
                    ),
                }
            )
            reset_verification()
            st.rerun()

        if contract:
            st.session_state.customer_contract = contract
            verification["stage"] = "ready_to_answer"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Policy verified successfully. I’ll answer your question now.",
                }
            )

            st.rerun()

        else:
            verification["stage"] = "waiting_policy_number"
            verification["policy_number"] = None

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The policy number or access code is incorrect. "
                        "Please enter your policy number again."
                    ),
                }
            )

            st.rerun()


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return {}


def route_message_with_gemini(user_message: str) -> dict:
    """
    Gemini is the main router.

    It decides which path the app should take:
    - small_talk: friendly conversation, greetings, thanks, jokes, etc.
    - general: company-level insurance information from uploaded documents / FAISS.
    - personal: customer's own policy, claim, payment, add-on, ticket, or account data from SQL.
    - both: needs both customer data and general policy documents.
    - out_of_scope: unrelated topic such as sports, weather, recipes, programming, etc.

    Important: Gemini only chooses the route. Python still controls verification,
    database access, FAISS retrieval, and safety.
    """

    router_prompt = f"""
You are the routing brain for SecureLife Assist, an insurance company chatbot.

Classify the user's message into exactly one route:

1. "small_talk"
Use this for greetings, thanks, goodbye, jokes, friendly chat, asking who you are,
asking how you are, or asking what you can do.
Examples:
- hi
- hey bot
- how are you?
- thank you
- tell me a joke
- i want your help
- what can you do?

2. "general"
Use this when the user asks about general SecureLife insurance information,
procedures, requirements, exclusions, complaints, renewals, cancellations, or claim steps.
This can be answered from uploaded company documents.
Examples:
- How do I file a complaint?
- How do I file a claim?
- What documents are required for a motor claim?
- What is excluded from travel insurance?
- How do I cancel my policy?

3. "personal"
Use this when the user asks about their own customer data, policy, claim status,
payment, premium, deductible, add-ons, support ticket, or contract.
This requires policy verification before answering.
Examples:
- What is my claim status?
- Did I pay my premium?
- When does my policy expire?
- Do I have roadside assistance?
- What is my deductible?

4. "both"
Use this when the user asks about their own policy plus general rules or procedures.
This requires verification and company documents.
Examples:
- Does my policy cover theft and what documents do I need?
- Am I covered for windshield damage and what is the claim process?
- Can I cancel my policy and how does the refund process work?

5. "out_of_scope"
Use this for topics not related to SecureLife insurance support.
Examples:
- Who won the football match?
- What is the weather?
- Write Python code
- Give me a recipe
- What is the capital of France?

Return only valid JSON in this exact format:
{{
  "route": "small_talk",
  "confidence": 0.95,
  "reason": "Short reason",
  "assistant_reply": "Only write a reply here for small_talk or out_of_scope. For general, personal, or both, leave this empty."
}}

Rules for assistant_reply:
- For "small_talk", respond naturally and warmly, then guide the user toward insurance topics.
- For "out_of_scope", respond briefly, do not pretend to have live updates, and guide the user back to claims, coverage, payments, renewals, complaints, or policy documents.
- For "general", "personal", and "both", assistant_reply must be an empty string.
- Do not ask for policy number or access code in assistant_reply. Python will handle verification.
- Do not answer insurance policy facts inside assistant_reply.

User message:
{user_message}
""".strip()

    response = get_llm(temperature=0.0, max_tokens=300).invoke(router_prompt)
    parsed = extract_json(message_to_text(response).strip())

    route = str(parsed.get("route", "out_of_scope")).lower().strip()

    valid_routes = ["small_talk", "general", "personal", "both", "out_of_scope"]
    if route not in valid_routes:
        route = "out_of_scope"

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    assistant_reply = str(parsed.get("assistant_reply", "") or "").strip()

    if route in ["general", "personal", "both"]:
        assistant_reply = ""

    if route == "small_talk" and not assistant_reply:
        assistant_reply = (
            "Hello! I’m SecureLife Assist. I can help with claims, coverage, payments, "
            "renewals, complaints, and policy documents. What would you like to check today?"
        )

    if route == "out_of_scope" and not assistant_reply:
        assistant_reply = (
            "I’m mainly here for SecureLife insurance support. I can help with claims, "
            "coverage, payments, renewals, complaints, or policy documents."
        )

    return {
        "route": route,
        "confidence": confidence,
        "reason": str(parsed.get("reason", "") or ""),
        "assistant_reply": assistant_reply,
    }


def build_general_prompt(rag_context: str, question: str) -> str:
    return f"""
You are SecureLife Assist, a professional digital customer support assistant for SecureLife Insurance.

Answer the customer's question using only the SecureLife information below.

Rules:
- Do not invent policy terms, coverage, exclusions, prices, or claim decisions.
- Do not mention internal sources, FAISS, retrieval, vector stores, database details, or system context.
- If the answer is not available, say exactly:
"{NOT_FOUND_MESSAGE}"
- Keep the answer clear, polite, and customer-friendly.
- Do not provide legal or financial advice.
- Do not guarantee claim approval.

SecureLife information:
{rag_context}

Customer question:
{question}

Answer:
""".strip()


def build_personal_prompt(customer_context: str, question: str) -> str:
    return f"""
You are SecureLife Assist, a professional digital customer support assistant for SecureLife Insurance.

The customer has been verified by the system. Use the customer data below to answer their question.

Rules:
- Answer only using the customer data provided.
- Decide which details are relevant to the question.
- Do not mention SQL, database tables, FAISS, retrieval, vector stores, internal sources, or system context.
- Do not reveal unnecessary personal information.
- Do not mention the access code.
- Do not invent claim status, payment status, coverage, add-ons, dates, or amounts.
- Do not start with "Dear".
- Do not write like a formal letter.
- Do not guarantee claim approval.
- If the data does not answer the question, say exactly:
"{NOT_FOUND_MESSAGE}"
- Keep the answer clear, polite, and customer-friendly.

Customer data:
{customer_context}

Customer question:
{question}

Answer:
""".strip()


def build_combined_prompt(customer_context: str, rag_context: str, question: str) -> str:
    return f"""
You are SecureLife Assist, a professional digital customer support assistant for SecureLife Insurance.

Use both sources below when needed:

1. Customer data:
This contains the verified customer's policy, claims, payments, add-ons, claim documents, and support tickets.

2. SecureLife policy information:
This contains general policy rules, exclusions, claim requirements, and procedures.

Rules:
- Use customer data for the customer's own policy, claim, payment, coverage, add-ons, documents, or support tickets.
- Use SecureLife policy information for general rules, required documents, exclusions, and procedures.
- Decide what is relevant to the customer's question.
- Do not mention SQL, database tables, FAISS, retrieval, vector stores, internal sources, or system context.
- Do not mention the access code.
- Do not reveal unnecessary personal information.
- Do not invent policy terms, prices, coverage, exclusions, claim status, or payment data.
- Do not start with "Dear".
- Do not write like a formal letter.
- Do not guarantee claim approval.
- If the answer is not available in either source, say exactly:
"{NOT_FOUND_MESSAGE}"
- Keep the answer clear, polite, and customer-friendly.

Customer data:
{customer_context}

SecureLife policy information:
{rag_context}

Customer question:
{question}

Answer:
""".strip()


def answer_from_general_documents(user_question: str) -> tuple[str, str]:
    if not knowledge_base_exists():
        answer = (
            "I could not find the SecureLife knowledge base yet. "
            "Please ask an administrator to upload and publish the company documents first. "
            f"For help, contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}."
        )
        return answer, "FAISS documents missing"

    rag_context = retrieve_rag_context(user_question, k=5)

    if not rag_context:
        return NOT_FOUND_MESSAGE, "FAISS documents empty"

    prompt = build_general_prompt(rag_context=rag_context, question=user_question)
    response = get_llm().invoke(prompt)
    answer = message_to_text(response)

    return answer, "FAISS documents"


def answer_from_customer_records(user_question: str) -> tuple[str, str]:
    contract = st.session_state.customer_contract

    if not contract:
        return NOT_FOUND_MESSAGE, "SQL customer records unavailable"

    customer_data = get_full_customer_data(contract["policy_number"])
    customer_context = build_full_customer_context(
        contract=contract,
        customer_data=customer_data,
    )

    prompt = build_personal_prompt(
        customer_context=customer_context,
        question=user_question,
    )

    response = get_llm().invoke(prompt)
    answer = message_to_text(response)

    return answer, "SQL customer records"


def answer_from_combined_sources(user_question: str) -> tuple[str, str]:
    contract = st.session_state.customer_contract

    if not contract:
        return NOT_FOUND_MESSAGE, "SQL customer records unavailable"

    customer_data = get_full_customer_data(contract["policy_number"])
    customer_context = build_full_customer_context(
        contract=contract,
        customer_data=customer_data,
    )

    rag_context = retrieve_rag_context(user_question, k=5)
    if not rag_context:
        rag_context = "No SecureLife document context is currently available."

    prompt = build_combined_prompt(
        customer_context=customer_context,
        rag_context=rag_context,
        question=user_question,
    )

    response = get_llm().invoke(prompt)
    answer = message_to_text(response)

    return answer, "SQL customer records + FAISS documents"


def answer_question(user_question: str, question_type: str, show_user_message: bool = True) -> None:
    if show_user_message:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner(get_spinner_text(question_type)):
            data_sources_used = "unknown"

            try:
                if question_type == "general":
                    answer, data_sources_used = answer_from_general_documents(user_question)

                elif question_type == "personal":
                    if not st.session_state.customer_contract:
                        start_verification(
                            question=user_question,
                            question_type="personal",
                        )
                        return

                    answer, data_sources_used = answer_from_customer_records(user_question)

                    # Optional fallback: if customer records do not answer, try policy documents.
                    if is_not_found_answer(answer) and knowledge_base_exists():
                        fallback_answer, fallback_source = answer_from_general_documents(user_question)
                        data_sources_used = f"{data_sources_used} → fallback to {fallback_source}"
                        answer = fallback_answer

                elif question_type == "both":
                    if not st.session_state.customer_contract:
                        start_verification(
                            question=user_question,
                            question_type="both",
                        )
                        return

                    answer, data_sources_used = answer_from_combined_sources(user_question)

                else:
                    answer = (
                        "I’m here to help with SecureLife insurance questions. "
                        "You can ask me about claims, coverage, payments, renewals, complaints, or required documents."
                    )
                    data_sources_used = "router fallback"

                if is_not_found_answer(answer):
                    answer = NOT_FOUND_MESSAGE

            except Exception as exc:
                if is_quota_error(exc):
                    answer = (
                        "SecureLife Assist has reached the current AI request limit. "
                        f"Please try again later or contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}."
                    )
                else:
                    answer = (
                        "Sorry, SecureLife Assist could not answer right now. "
                        f"Please contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}. "
                        f"Details: {exc}"
                    )

                data_sources_used = "error"

        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        contract = st.session_state.customer_contract

        if contract:
            try:
                insert_audit_log(
                    customer_id=contract["customer_id"],
                    policy_number=contract["policy_number"],
                    question=user_question,
                    answer=answer,
                    data_sources_used=data_sources_used,
                )
            except Exception:
                pass


st.title(APP_NAME)

st.caption(
    "Your digital insurance support assistant for policy information, claims guidance, "
    "coverage details, renewals, cancellations, and customer service questions."
)

st.info(
    "SecureLife Assist is designed to support customers with insurance questions using approved company knowledge. "
    "You can ask about coverage, claims, required documents, payments, renewals, cancellations, and complaints. "

)

if not api_key_exists():
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
    st.stop()

with st.sidebar:
    st.header("Example Questions")

    examples = [
        "How do I file a complaint?",
        "What documents are required for a motor claim?",
        "What is excluded from travel insurance?",
        "What is my claim status?",
        "Do I have roadside assistance?",
        "Does my policy cover theft and what documents do I need?",
    ]

    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.example_question = example
            st.rerun()

    st.divider()

    if st.session_state.customer_contract:
        st.success(f"Verified policy: {st.session_state.customer_contract['policy_number']}")

    if st.button("Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.customer_contract = None
        reset_verification()
        st.session_state.example_question = None
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

verification = st.session_state.verification

if verification["stage"] == "ready_to_answer":
    original_question = verification["question"]
    original_question_type = verification["question_type"]

    reset_verification()

    answer_question(
        user_question=original_question,
        question_type=original_question_type,
        show_user_message=False,
    )

chat_message = st.chat_input("How can we help you with your insurance today?")

if st.session_state.example_question:
    user_message = st.session_state.example_question
    st.session_state.example_question = None
else:
    user_message = chat_message

if user_message:
    verification = st.session_state.verification

    if verification["stage"] == "waiting_policy_number":
        if looks_like_policy_number(user_message):
            handle_verification_message(user_message)
            st.stop()

        # User changed subject instead of entering policy number
        cancel_verification_and_continue()

    elif verification["stage"] == "waiting_access_code":
        if looks_like_access_code(user_message):
            handle_verification_message(user_message)
            st.stop()

        # User changed subject instead of entering access code
        cancel_verification_and_continue()

    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.write(user_message)

    # ---------------------------------------------------------
    # Gemini router
    # ---------------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Understanding your message..."):
            try:
                route_result = route_message_with_gemini(user_message)

            except Exception as exc:
                if is_quota_error(exc):
                    answer = (
                        "SecureLife Assist has reached the current AI request limit. "
                        f"Please try again later or contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}."
                    )
                else:
                    answer = (
                        "Sorry, SecureLife Assist could not understand the request right now. "
                        f"Please contact SecureLife customer support at {CUSTOMER_SUPPORT_PHONE}. "
                        f"Details: {exc}"
                    )

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

        route = route_result["route"]

        if route in ["small_talk", "out_of_scope"]:
            answer = route_result["assistant_reply"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

    # For insurance-related routes, Python executes the route Gemini selected.
    answer_question(
        user_question=user_message,
        question_type=route,
        show_user_message=False,
    )
