import json
import re
import numpy as np
import requests
from elasticsearch import Elasticsearch

# Safely initialize the client
try:
    # Ensure Elastic is running on port 9200
    es = Elasticsearch("http://localhost:9200")
    es_connected = es.ping()
except Exception as e:
    print(f"Elasticsearch connection failed: {e}")
    es_connected = False

def store_to_elastic(data):
    """Stores ticket embeddings in Elasticsearch."""
    if es_connected:
        es.index(index="support_tickets", document=data)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Secure MNC AI Semantic Routing Engine", version="1.1.0")

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

ROUTES = {
    "Cloud infrastructure, servers, and DevOps": "cloud-ops-team",
    "Billing, invoices, payments, and refunds": "finance-support",
    "Access management, passwords, and security IAM": "secops-tier1"
}

class TicketPayload(BaseModel):
    ticket_text: str

def mask_pii(text: str) -> str:
    """Enterprise compliance layer: Scrubber for common PII patterns."""
    # Mask Email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    # Mask Credit Cards (Basic 16 digit pattern)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED_CARD]', text)
    # Mask IPv4 Addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]', text)
    return text

def get_local_embedding(text: str):
    try:
        response = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
        return np.array(response.json()["embedding"])
    except Exception:
        return np.zeros(768)

def cosine_similarity(v1, v2):
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (mag1 * mag2)) if mag1 and mag2 else 0.0

print("Precomputing secure semantic routing maps...")
ROUTE_EMBEDDINGS = {route: get_local_embedding(route) for route in ROUTES.keys()}

@app.post("/route")
def handle_routing_request(payload: TicketPayload):
    # Enforce scrubbing immediately upon ingestion
    sanitized_text = mask_pii(payload.ticket_text)
    
    if not sanitized_text.strip():
        raise HTTPException(status_code=400, detail="Ticket payload text string cannot be empty.")
        
    ticket_vector = get_local_embedding(sanitized_text)
    best_route, highest_score = None, -1.0

    for route, route_vector in ROUTE_EMBEDDINGS.items():
        score = cosine_similarity(ticket_vector, route_vector)
        if score > highest_score:
            highest_score, best_route = score, route

    if highest_score >= 0.50:
        return {
            "assigned_queue": ROUTES[best_route],
            "routing_mechanism": "Tier1_Vector_Similarity",
            "confidence_score": round(highest_score, 4),
            "processed_text_preview": sanitized_text[:100]  # Proves data was successfully sanitized
        }

    # Tier 2: Local LLM Fallback
    system_prompt = (
        "You are an enterprise ticket routing AI. Analyze the ticket and return a JSON object with keys: "
        "'assigned_queue' (must be one of: 'cloud-ops-team', 'finance-support', 'secops-tier1', 'manual-triage-queue'), "
        "'urgency' ('P1', 'P2', 'P3'), and 'reasoning'."
    )
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": f"{system_prompt}\n\nTicket: {sanitized_text}\n\nReturn strict raw JSON:", "stream": False, "format": "json"}
        )
        decision = json.loads(response.json()["response"])
        decision["routing_mechanism"] = "Tier2_LLM_Reasoning"
        decision["processed_text_preview"] = sanitized_text[:100]
        return decision
    except Exception as e:
        return {"assigned_queue": "manual-triage-queue", "routing_mechanism": "Tier2_Pipeline_Error", "reasoning": str(e)}
