import json
import os
import re
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

app = FastAPI(title="Commercial AI Routing SaaS Engine", version="2.0.0")

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

CLIENT_DATABASE = {
    "client_alpha_secret_key": {"name": "Alpha Corp", "used_requests": 0, "max_quota": 1000},
    "client_beta_secret_key": {"name": "Beta Logistics", "used_requests": 0, "max_quota": 5}
}

ROUTES = {
    "Cloud infrastructure, servers, and DevOps": "cloud-ops-team",
    "Billing, invoices, payments, and refunds": "finance-support",
    "Access management, passwords, and security IAM": "secops-tier1"
}

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class TicketPayload(BaseModel):
    ticket_text: str

def verify_client(api_key: str = Depends(api_key_header)):
    if not api_key or api_key not in CLIENT_DATABASE:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid or Missing SaaS API Key.")
    
    client = CLIENT_DATABASE[api_key]
    if client["used_requests"] >= client["max_quota"]:
        raise HTTPException(status_code=429, detail="Rate Limit Exceeded: Monthly quota exhausted.")
    
    return api_key

def mask_pii(text: str) -> str:
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED_CARD]', text)
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

print("SaaS Engine Initialized. Mapping vectors...")
ROUTE_EMBEDDINGS = {route: get_local_embedding(route) for route in ROUTES.keys()}

@app.post("/api/v1/route")
def handle_saas_routing(payload: TicketPayload, api_key: str = Depends(verify_client)):
    CLIENT_DATABASE[api_key]["used_requests"] += 1
    current_client = CLIENT_DATABASE[api_key]

    sanitized_text = mask_pii(payload.ticket_text)
    ticket_vector = get_local_embedding(sanitized_text)
    best_route, highest_score = None, -1.0

    for route, route_vector in ROUTE_EMBEDDINGS.items():
        score = cosine_similarity(ticket_vector, route_vector)
        if score > highest_score:
            highest_score, best_route = score, route

    if highest_score >= 0.50:
        return {
            "success": True,
            "assigned_queue": ROUTES[best_route],
            "billing_info": {
                "client": current_client["name"],
                "requests_used_this_month": current_client["used_requests"],
                "remaining_quota": current_client["max_quota"] - current_client["used_requests"]
            },
            "routing_metrics": {"mechanism": "Tier1_Vector", "confidence": round(highest_score, 4)}
        }

    system_prompt = "Analyze the ticket and return JSON keys: 'assigned_queue' (cloud-ops-team, finance-support, secops-tier1), 'urgency'."
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": f"{system_prompt}\n\nTicket: {sanitized_text}\n\nReturn JSON:", "stream": False, "format": "json"}
        )
        decision = json.loads(response.json()["response"])
        return {
            "success": True,
            "assigned_queue": decision.get("assigned_queue", "manual-triage-queue"),
            "billing_info": {
                "client": current_client["name"],
                "requests_used_this_month": current_client["used_requests"],
                "remaining_quota": current_client["max_quota"] - current_client["used_requests"]
            },
            "routing_metrics": {"mechanism": "Tier2_LLM_Reasoning", "urgency": decision.get("urgency", "P3")}
        }
    except Exception:
        return {"success": False, "assigned_queue": "manual-triage-queue", "error": "Internal computation error"}
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def render_client_portal():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MNC Secure AI Router - Client Portal</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f3f4f6; color: #1f2937; padding: 40px; margin: 0; }
            .container { max-width: 650px; background: white; margin: 0 auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h2 { color: #111827; margin-top: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
            label { font-weight: 600; display: block; margin: 20px 0 8px; font-size: 14px; }
            input[type="text"], textarea { width: 100%; padding: 12px; border: 1px solid #d1d5db; border-radius: 6px; box-sizing: border-box; font-size: 15px; }
            textarea { height: 120px; resize: none; }
            button { width: 100%; background: #2563eb; color: white; border: none; padding: 14px; font-size: 16px; font-weight: 600; border-radius: 6px; cursor: pointer; margin-top: 20px; transition: background 0.2s; }
            button:hover { background: #1d4ed8; }
            .result-box { margin-top: 25px; padding: 20px; border-radius: 8px; display: none; border-left: 5px solid; }
            .success-box { background-color: #f0fdf4; border-color: #22c55e; color: #166534; }
            .error-box { background-color: #fef2f2; border-color: #ef4444; color: #991b1b; }
            pre { margin: 5px 0 0; font-family: monospace; background: rgba(0,0,0,0.04); padding: 10px; border-radius: 4px; font-size: 13px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>MNC AI Semantic Routing Gateway</h2>
            <p style="font-size: 14px; color: #6b7280; margin-bottom: 20px;">Submit raw customer inquiries securely. Our local vector network strips PII automatically and assigns target support queues in real time.</p>
            
            <label for="apiKey">Your Client Subscription Key (X-API-KEY)</label>
            <input type="text" id="apiKey" placeholder="Enter your secret SaaS token..." value="client_alpha_secret_key">
            
            <label for="ticketText">Support Ticket Description Text</label>
            <textarea id="ticketText" placeholder="Example: Our payment processing page threw an error during check out..."></textarea>
            
            <button onclick="submitTicketRoute()">Execute Semantic Triage</button>
            
            <div id="resultBox" class="result-box">
                <strong id="statusTitle">Routing Match Successful!</strong>
                <p style="margin: 8px 0 4px; font-size: 14px;"><strong>Target Queue:</strong> <span id="queueOutput"></span></p>
                <p style="margin: 0 0 10px; font-size: 14px;"><strong>Billing Analytics:</strong> <span id="billingOutput"></span></p>
                <pre id="jsonRaw"></pre>
            </div>
        </div>

        <script>
            async function submitTicketRoute() {
                const apiKey = document.getElementById('apiKey').value;
                const ticketText = document.getElementById('ticketText').value;
                const resultBox = document.getElementById('resultBox');
                
                if (!ticketText.trim()) { alert('Please enter support ticket text.'); return; }
                
                try {
                    const response = await fetch('/api/v1/route', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'X-API-KEY': apiKey },
                        body: JSON.stringify({ ticket_text: ticketText })
                    });
                    
                    const data = await response.json();
                    resultBox.style.display = 'block';
                    
                    if (response.ok) {
                        resultBox.className = 'result-box success-box';
                        document.getElementById('statusTitle').innerText = 'Routing Execution Successful!';
                        document.getElementById('queueOutput').innerText = data.assigned_queue.toUpperCase();
                        document.getElementById('billingOutput').innerText = `Used: ${data.billing_info.requests_used_this_month} requests | Remainder: ${data.billing_info.remaining_quota}`;
                        document.getElementById('jsonRaw').innerText = JSON.stringify(data.routing_metrics || data, null, 2);
                    } else {
                        resultBox.className = 'result-box error-box';
                        document.getElementById('statusTitle').innerText = 'Authentication / Pipeline Failure';
                        document.getElementById('queueOutput').innerText = 'NONE';
                        document.getElementById('billingOutput').innerText = 'N/A';
                        document.getElementById('jsonRaw').innerText = JSON.stringify(data, null, 2);
                    }
                } catch (err) {
                    alert('Could not link to local microservice.');
                }
            }
        </script>
    </body>
    </html>
    """
