# SecureLife Assist 🛡️

SecureLife Assist is an AI-powered insurance customer support chatbot built with Streamlit, Gemini, FAISS, LangChain, and PostgreSQL/Supabase.

The project simulates a real-life insurance company assistant that can answer general policy questions from approved company documents and answer personal customer questions after policy verification.

---

## Overview

Insurance companies receive many repetitive customer questions about claims, coverage, required documents, payments, renewals, cancellations, and complaints. SecureLife Assist demonstrates how an AI assistant can support these workflows by combining:

- Retrieval-Augmented Generation using company policy documents
- Gemini-based message routing
- SQL database access for customer-specific records
- Policy number and access code verification
- Admin document upload for updating the knowledge base
- Safety rules to avoid hallucinated policy or claim decisions

This project is built as a realistic AI engineering portfolio project, not just a simple chatbot demo.

---

## Key Features

### Customer Chatbot

Customers can ask questions such as:

- How do I file a complaint?
- What documents are required for a motor claim?
- What is excluded from motor insurance?
- What is my claim status?
- Do I have roadside assistance?
- Does my policy cover theft and what documents do I need?

The chatbot can handle:

- General insurance questions
- Personal policy questions
- Questions that require both policy documents and customer records
- Small talk
- Out-of-scope questions

---

### Gemini Router

Gemini is used as the routing brain of the application. It classifies each user message into one of the following routes:

- `small_talk`
- `general`
- `personal`
- `both`
- `out_of_scope`

Python then controls the actual system behavior, including verification, database access, FAISS retrieval, and response generation.

This keeps the system more reliable because Gemini decides the intent, but the application code controls sensitive operations.

---

### RAG Knowledge Base

General insurance questions are answered using a FAISS vector database built from approved company documents.

The knowledge base can include:

- Policy wording
- Claim process guides
- FAQ documents
- Complaint procedures
- Renewal and cancellation rules
- Service information

The assistant only answers from the retrieved company knowledge and is instructed not to invent policy terms, coverage, prices, exclusions, or claim decisions.

---

### Personal Customer Data

For personal questions, the chatbot asks the user to verify their policy first using:

- Policy number
- Access code

After successful verification, the app can answer questions using customer records from the SQL database, including:

- Policy details
- Add-ons
- Claims
- Claim documents
- Payments
- Support tickets

The assistant does not expose unnecessary personal data and does not mention internal database details to the user.

---

### Admin Portal

The Admin Portal allows company staff to manage the chatbot knowledge base.

Admins can:

- Upload approved `.txt` or `.csv` documents
- Build a FAISS vector index
- Replace old knowledge base content
- Publish updated company documents for the chatbot

For this demo version, the Admin Portal is open locally. If this project is deployed publicly, authentication should be added before exposing the Admin Portal.

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Gemini API
- FAISS
- SQLAlchemy
- PostgreSQL / Supabase
- python-dotenv

---

## Project Structure

```text
securelife-assist/
│
├── streamlit_app.py
├── main.py
├── admin.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   └── sample_documents/
│       ├── claims_process_faq.txt
│       ├── company_profile.txt
│       ├── complaints_escalation.txt
│       ├── motor_comprehensive_policy.txt
│       └── renewals_cancellations_refunds.txt
│
└── screenshots/
    ├── chatbot.png
    └── admin_portal.png
