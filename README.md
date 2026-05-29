# Local-First B2B AI Semantic Routing Gateway

A high-performance, cost-effective API gateway that secure-routes client support tickets using offline vector embeddings.

## 🚀 Key Architectural Features
- **Local AI Context Extraction**: Uses `nomic-embed-text` and `llama3` running locally via Ollama to eliminate cloud inference fees.
- **B2B Subscription Middleware**: Authenticates requests via custom `X-API-KEY` tokens and enforces usage rate-limits dynamically.
- **Embedded Web Client Portal**: Features an integrated asynchronous HTML/JS dashboard interface for prompt triage testing.
- **Linux Systemd Resilience**: Configured as a native background service utility layer for automated crash recovery loops.

## 🛠️ Technical Stack
- **Backend Core**: Python, FastAPI, Uvicorn, Pydantic
- **Vector Network Engine**: Ollama, NumPy Vector Matching
- **Ingress Gateway Routing**: Ngrok Encrypted Tunneling
