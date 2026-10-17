import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import (PIIMiddleware,before_model,HumanInTheLoopMiddleware,after_model)
from transformers import pipeline
from langchain_core.messages import HumanMessage
from langgraph.checkpointer import InMemorySaver 

load_dotenv()

model = init_chat_model("google_genai:gemini-3.1-flash-lite", api_key=os.environ["GEMINI_API_KEY"])


INBOX = [
    {
        "id": 1,
        "from": "maria@example.com",
        "subject": "Claim Status - POL-48120375",
        "body": "Any update on my motor claim filed last week?",
    },
    {
        "id": 2,
        "from": "offers@spam.io",
        "subject": "LOWEST PREMIUM EVER",
        "body": "Switch now and save big, click here...",
    },
]


@tool
def search_emails(query: str) -> list[dict]:
    """Search the inbox by sender, subject, or keyword."""
    q = query.lower()
    return [
        email
        for email in INBOX
        if q in (email["from"] + email["subject"] + email["body"]).lower()
    ]


@tool
def draft_reply(email_id: int, body: str) -> str:
    """Draft a reply to a specific email. Does not send."""
    return f"Draft saved for email {email_id}: {body}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This action is irreversible."""
    return f"Email sent to {to} | Subject: {subject}"

pii_layer = [
    PIIMiddleware("email", strategy="redact"),
    PIIMiddleware("credit_card", strategy="mask"),
    PIIMiddleware("policy_number", strategy="mask", detector=r"POL-\d{8}")
]

injection_clf = pipeline(
    "text-classification",
    model="protectai/deberta-v3-base-prompt-injection-v2",
)

@before_model 
def block_injections(state,runtime):
    last_prompt = state["messages"][-1].text 
    if isinstance(last_prompt, HumanMessage):
        result = injection_clf(last_prompt)[0]
        if result['label'] == "INJECTION" and result["score"] > 0.8:
            raise ValueError(f"Blocked dur to Malicious Prompt ({result['score']:.2f})")

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "allowed_decisions": ["approve", "edit", "reject"]
        }
    }
)

@after_model
def block_settlement_commitments(state,runtime):
    last_reply = state["messages"][-1].text.lower()
    forbidden_phrases = [
        "we will refund",
        "your refund of",
        "refund of $",
        "refund of rs"
    ]
    for phase in forbidden_phrases:
        if last_reply in phase:
            raise ValueError("Blocked: reply commits to a refund — requires manager approval")

# Write your code here

system_prompt = """
You are a helpful email assistant for an insurance company's claims desk.

You have access to tools for searching emails, drafting replies, and sending emails.
When a user asks for an email-related task, use the relevant tool to complete it.
Choose the right tool based on the user's request.
"""

agent = create_agent(
    model=model,
    tools=[search_emails, draft_reply, send_email],
    system_prompt=system_prompt,
    middleware=[block_injections,*pii_layer,hitl,block_settlement_commitments],
    checkpointer = InMemorySaver()
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Reply to maria@example.com about her claim status. Tell her the review is in progress.",
            }
        ]
    }
)
print(response["messages"][-1].text)
