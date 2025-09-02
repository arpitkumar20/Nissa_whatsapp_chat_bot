# Enterprise Folder Structure for Nisaa Chatbot

```
nisaa-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   │
│   │   ├── api/                     # REST & Webhook endpoints
│   │   │   ├── routes_chat.py       # Chat/Conversation endpoints
│   │   │   ├── routes_whatsapp.py   # Twilio/Meta webhook handlers
│   │   │   ├── routes_admin.py      # Admin panel APIs
│   │   │   └── routes_booking.py    # HMS / Hotel booking endpoints
│   │   │
│   │   ├── core/                    # System-wide configs & middleware
│   │   │   ├── config.py
│   │   │   ├── logging_conf.py
│   │   │   └── security.py
│   │   │
│   │   ├── services/                # Business logic
│   │   │   ├── rag.py               # Retrieval-Augmented Generation pipeline
│   │   │   ├── vectorstore.py       # Vector DB management (FAISS / Pinecone)
│   │   │   ├── Scrapper.py          # Website crawling & content extraction
│   │   │   ├── ingestion.py         # Indexing scraped data → vector DB
│   │   │   ├── whatsapp.py          # WhatsApp/Twilio integration logic
│   │   │   ├── hms_integration.py   # Hospital system API integration
│   │   │   ├── hotel_integration.py # Hotel booking API integration
│   │   │   └── analytics.py         # Metrics & chatbot usage analytics
│   │   │
│   │   ├── models/                  # Domain models
│   │   │   ├── user.py
│   │   │   ├── chat_session.py
│   │   │   ├── booking.py
│   │   │   └── doctor.py
│   │   │
│   │   ├── utils/                   # Helper functions
│   │   │   ├── nlp_utils.py
│   │   │   ├── formatters.py
│   │   │   └── validators.py
│   │   │
│   │   ├── admin/                   # Admin panel logic
│   │   │   ├── dashboard.py
│   │   │   ├── templates_manager.py
│   │   │   └── lead_tracking.py
│   │   │
│   │   └── auth/                    # Authentication & access control
│   │       ├── api_key.py
│   │       └── oauth.py
│   │
│   ├── tests/                       # Testing layers
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── performance/
│   │   └── security/
│   │
│   ├── scripts/                     # Utility scripts
│   │   ├── seed_data.py
│   │   ├── migrate_db.py
│   │   └── load_templates.py
│   │
│   ├── Dockerfile
│   ├── requirements.txt
│   └── wsgi.py
│
├── frontend/                        # Optional: React/Next.js Admin Panel
│   ├── components/
│   ├── pages/
│   ├── services/
│   └── package.json
│
├── infra/                           # Deployment, monitoring, DevOps
│   ├── docker-compose.yml
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   ├── monitoring/
│   │   ├── prometheus.yml
│   │   └── grafana-dashboard.json
│   └── ci-cd/
│       └── github-actions.yml
│
├── docs/                            # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── testing_strategy.md
│
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## How This Maps to Features

* **Conversational Intelligence (Q\&A, RAG)** → `services/rag.py`, `services/vectorstore.py`, `api/routes_chat.py`
* **Website Crawling & Indexing** → `services/crawler.py`, `services/ingestion.py`
* **HMS & Booking** → `services/hms_integration.py`, `services/hotel_integration.py`, `models/booking.py`
* **WhatsApp** → `services/whatsapp.py`, `api/routes_whatsapp.py`
* **Admin Panel** → `admin/`, `api/routes_admin.py`, frontend dashboard
* **360° Room Tours / Hotel Enhancements** → `services/hotel_integration.py` + frontend UI
* **Hospital Features** → `models/doctor.py`, `services/hms_integration.py`
* **DevOps/Infra** → `infra/` (Docker, K8s, CI/CD, monitoring)
* **Testing** → `tests/unit`, `tests/integration`, `tests/performance`, `tests/security`

