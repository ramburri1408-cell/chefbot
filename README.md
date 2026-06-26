# ChefBot — AI Menu Assistant

Production-ready conversational food recommendation engine built with LangGraph, RAG, and the Anthropic Claude API. Designed to scale to millions of concurrent restaurant sessions.

## Architecture

```
Client (Next.js)
    │  WebSocket (streaming tokens)
    ▼
API Gateway (FastAPI)
    │  Auth middleware · Rate limiting · Session hydration
    ├──► LangGraph Agent  ──► RAG Service ──► Vector Store (Chroma/Pinecone)
    └──► Menu Ingestion   ──► PostgreSQL
         │
         └──► Anthropic Claude API (intent extraction + generation)
```

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI + WebSocket | Streaming responses, async I/O |
| Agent | LangGraph | Explicit stateful multi-node graph |
| RAG | ChromaDB (dev) / Pinecone (prod) | Semantic dish retrieval |
| DB | PostgreSQL + SQLAlchemy | Menu data, feedback, analytics |
| Cache | Redis | Session state, rate limiting |
| LLM | Anthropic Claude claude-sonnet-4-6 | Intent + generation |
| Frontend | Next.js 14 + TypeScript | SSR, WebSocket, streaming UI |
| Infra | Docker + Kubernetes | Horizontal scaling |

## Local setup

```bash
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, DATABASE_URL, REDIS_URL

docker-compose up -d        # starts postgres + redis
cd backend
pip install -r requirements.txt
alembic upgrade head        # run migrations
python -m app.ingestion     # index menu into vector store
uvicorn app.main:app --reload

cd ../frontend
npm install
npm run dev
```

## Project structure

```
chefbot/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry, lifespan, middleware
│   │   ├── api/
│   │   │   ├── chat.py          # WebSocket endpoint
│   │   │   └── health.py        # /health, /ready
│   │   ├── core/
│   │   │   ├── config.py        # Settings (pydantic-settings)
│   │   │   ├── security.py      # JWT, rate limiting
│   │   │   └── logging.py       # Structured JSON logs
│   │   ├── services/
│   │   │   ├── agent.py         # LangGraph graph definition
│   │   │   ├── rag.py           # Vector store + retrieval
│   │   │   └── ingestion.py     # Menu → embeddings → store
│   │   ├── models/
│   │   │   ├── menu.py          # SQLAlchemy ORM models
│   │   │   └── session.py       # Redis session schema
│   │   └── db/
│   │       ├── postgres.py      # Engine + session factory
│   │       └── redis.py         # Redis client
│   ├── migrations/              # Alembic migrations
│   ├── data/
│   │   └── menu.json            # Seed data
│   ├── tests/
│   │   ├── test_agent.py
│   │   └── test_rag.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx         # Main chat UI
│   │   │   ├── Message.tsx      # Individual message + dish cards
│   │   │   └── DishCard.tsx     # Nutrition card component
│   │   ├── hooks/
│   │   │   └── useChat.ts       # WebSocket + streaming hook
│   │   ├── lib/
│   │   │   └── ws.ts            # WebSocket client with reconnect
│   │   └── types/
│   │       └── index.ts         # Shared TypeScript types
│   ├── public/
│   └── package.json
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.backend
│   │   └── Dockerfile.frontend
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
├── docker-compose.yml
└── .env.example
```
