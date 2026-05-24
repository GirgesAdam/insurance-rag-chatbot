import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
_engine = None


def get_engine():
    global _engine

    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL is missing. Add it to your .env file."
        )

    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
        )

    return _engine


def authenticate_customer(policy_number: str, access_code: str) -> Optional[dict]:
    query = text("""
        SELECT
            c.customer_id,
            c.full_name,
            c.email,
            c.phone,
            c.city,
            p.policy_number,
            p.product_name,
            p.policy_type,
            p.start_date,
            p.end_date,
            p.status,
            p.premium,
            p.deductible,
            p.coverage_limit,
            p.vehicle_make,
            p.vehicle_model,
            p.vehicle_year,
            p.plate_number
        FROM customers c
        JOIN policies p ON c.customer_id = p.customer_id
        WHERE p.policy_number = :policy_number
        AND c.access_code = :access_code
    """)

    engine = get_engine()

    with engine.connect() as conn:
        row = conn.execute(
            query,
            {
                "policy_number": policy_number,
                "access_code": access_code,
            },
        ).mappings().fetchone()

    return dict(row) if row else None


def get_policy_addons(policy_number: str) -> list[dict]:
    query = text("""
        SELECT addon_name, addon_status, effective_date
        FROM policy_addons
        WHERE policy_number = :policy_number
        ORDER BY addon_name
    """)

    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"policy_number": policy_number},
        ).mappings().fetchall()

    return [dict(row) for row in rows]


def get_customer_claims(policy_number: str) -> list[dict]:
    query = text("""
        SELECT
            claim_id,
            claim_type,
            claim_status,
            submitted_date,
            estimated_resolution_date,
            last_update,
            claim_amount,
            approved_amount,
            updated_at
        FROM claims
        WHERE policy_number = :policy_number
        ORDER BY submitted_date DESC
    """)

    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"policy_number": policy_number},
        ).mappings().fetchall()

    return [dict(row) for row in rows]


def get_customer_payments(policy_number: str) -> list[dict]:
    query = text("""
        SELECT payment_id, amount, due_date, status, paid_at
        FROM payments
        WHERE policy_number = :policy_number
        ORDER BY due_date DESC
    """)

    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"policy_number": policy_number},
        ).mappings().fetchall()

    return [dict(row) for row in rows]


def get_support_tickets(policy_number: str) -> list[dict]:
    query = text("""
        SELECT ticket_id, subject, status, priority, created_at, last_update
        FROM support_tickets
        WHERE policy_number = :policy_number
        ORDER BY created_at DESC
    """)

    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"policy_number": policy_number},
        ).mappings().fetchall()

    return [dict(row) for row in rows]


def get_claim_documents(policy_number: str) -> list[dict]:
    query = text("""
        SELECT
            cd.document_id,
            cd.claim_id,
            cd.document_name,
            cd.document_status,
            cd.uploaded_at
        FROM claim_documents cd
        JOIN claims c ON cd.claim_id = c.claim_id
        WHERE c.policy_number = :policy_number
        ORDER BY cd.claim_id, cd.document_name
    """)

    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {"policy_number": policy_number},
        ).mappings().fetchall()

    return [dict(row) for row in rows]


def get_full_customer_data(policy_number: str) -> dict:
    return {
        "addons": get_policy_addons(policy_number),
        "claims": get_customer_claims(policy_number),
        "payments": get_customer_payments(policy_number),
        "tickets": get_support_tickets(policy_number),
        "claim_documents": get_claim_documents(policy_number),
    }


def build_full_customer_context(contract: dict, customer_data: dict) -> str:
    addons = customer_data.get("addons", [])
    claims = customer_data.get("claims", [])
    payments = customer_data.get("payments", [])
    tickets = customer_data.get("tickets", [])
    claim_documents = customer_data.get("claim_documents", [])

    addons_text = "\n".join(
        f"- {addon.get('addon_name')}: {addon.get('addon_status')}, "
        f"effective date: {addon.get('effective_date')}"
        for addon in addons
    ) or "No add-ons found."

    claims_text = "\n".join(
        f"- Claim {claim.get('claim_id')}: {claim.get('claim_type')}, "
        f"status: {claim.get('claim_status')}, "
        f"submitted: {claim.get('submitted_date')}, "
        f"estimated resolution: {claim.get('estimated_resolution_date')}, "
        f"claim amount: {claim.get('claim_amount')}, "
        f"approved amount: {claim.get('approved_amount')}, "
        f"last update: {claim.get('last_update')}"
        for claim in claims
    ) or "No claims found."

    payments_text = "\n".join(
        f"- Payment {payment.get('payment_id')}: amount {payment.get('amount')}, "
        f"due date: {payment.get('due_date')}, "
        f"status: {payment.get('status')}, "
        f"paid at: {payment.get('paid_at')}"
        for payment in payments
    ) or "No payments found."

    tickets_text = "\n".join(
        f"- Ticket {ticket.get('ticket_id')}: {ticket.get('subject')}, "
        f"status: {ticket.get('status')}, "
        f"priority: {ticket.get('priority')}, "
        f"created at: {ticket.get('created_at')}, "
        f"last update: {ticket.get('last_update')}"
        for ticket in tickets
    ) or "No support tickets found."

    claim_documents_text = "\n".join(
        f"- Document {doc.get('document_id')} for claim {doc.get('claim_id')}: "
        f"{doc.get('document_name')}, "
        f"status: {doc.get('document_status')}, "
        f"uploaded at: {doc.get('uploaded_at')}"
        for doc in claim_documents
    ) or "No claim document records found."

    vehicle_text = ""
    if contract.get("policy_type") == "Motor":
        vehicle_text = f"""
Vehicle Information:
- Make: {contract.get('vehicle_make')}
- Model: {contract.get('vehicle_model')}
- Year: {contract.get('vehicle_year')}
- Plate number: {contract.get('plate_number')}
""".strip()

    return f"""
Authenticated Customer:
- Customer ID: {contract.get('customer_id')}
- Name: {contract.get('full_name')}
- City: {contract.get('city')}

Policy Information:
- Policy number: {contract.get('policy_number')}
- Product name: {contract.get('product_name')}
- Policy type: {contract.get('policy_type')}
- Policy status: {contract.get('status')}
- Start date: {contract.get('start_date')}
- End date: {contract.get('end_date')}
- Premium: {contract.get('premium')}
- Deductible: {contract.get('deductible')}
- Coverage limit: {contract.get('coverage_limit')}

{vehicle_text}

Policy Add-ons:
{addons_text}

Claims:
{claims_text}

Claim Documents:
{claim_documents_text}

Payments:
{payments_text}

Support Tickets:
{tickets_text}
""".strip()


def insert_audit_log(
    customer_id: str,
    policy_number: str,
    question: str,
    answer: str,
    data_sources_used: str,
) -> None:
    query = text("""
        INSERT INTO audit_logs
        (customer_id, policy_number, question, answer, data_sources_used)
        VALUES
        (:customer_id, :policy_number, :question, :answer, :data_sources_used)
    """)

    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(
            query,
            {
                "customer_id": customer_id,
                "policy_number": policy_number,
                "question": question,
                "answer": answer,
                "data_sources_used": data_sources_used,
            },
        )