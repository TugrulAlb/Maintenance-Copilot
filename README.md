# Maintenance Copilot

Maintenance Copilot is a portfolio project for semantic and analytical question answering over industrial maintenance and fault logs.

## What is here

- `data/generate_synthetic_data.py` creates a SQLite database with realistic maintenance tickets and LLM-generated technician notes.
- `ingestion/` chunks SQLite maintenance records and stores embeddings plus metadata in ChromaDB.
- `retrieval/` combines vector search and BM25 with metadata filtering and reranking.
- `graph/` contains the compiled LangGraph workflow with analytical, semantic, and hybrid branches.
- `api/` exposes the browser UI, API-key auth, role checks, rate limiting, CORS, and metrics.
- `evaluation/` runs regression checks for routing, graph nodes, citations, and answer content.
- `docker/`, `k8s/`, and `.github/workflows/` are reserved for deployment assets.

## Synthetic data generation

The data generator uses an OpenAI-compatible client to batch-generate natural language fault descriptions so later semantic search tests have real linguistic variety to work with. Structured fields are generated locally to keep the dataset deterministic and easy to inspect.

Run it after setting environment variables from `.env.example`:

```bash
python data/generate_synthetic_data.py --records 500
```

## Run the API locally

Start the FastAPI server from the project root:

```bash
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
	-H "Content-Type: application/json" \
	-d '{"question":"Motor failure nedeniyle duran hatlar hangileri?"}'
```

Ask with API-key auth enabled:

```bash
MAINTENANCE_COPILOT_ALLOW_PUBLIC=false \
MAINTENANCE_COPILOT_API_KEYS=demo-user-key,demo-admin-key \
MAINTENANCE_COPILOT_ADMIN_KEYS=demo-admin-key \
uvicorn api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
	-H "Content-Type: application/json" \
	-H "X-API-Key: demo-user-key" \
	-d '{"question":"Line 2 motor failure trendini ve benzer kayıtları özetle"}'
```

Metrics require an admin key when auth is enabled:

```bash
curl -H "X-API-Key: demo-admin-key" http://127.0.0.1:8000/metrics
```

Ask with an existing thread id:

```bash
curl -X POST http://127.0.0.1:8000/ask \
	-H "Content-Type: application/json" \
	-d '{"question":"Sensör hatası olan makineler","thread_id":"demo-thread-1"}'
```

## Run with Docker

Build and start the API container:

```bash
cd docker
docker compose up --build
```

The Docker image contains the application code and Python dependencies. The SQLite
database and ChromaDB storage are mounted as volumes so they persist across restarts
and can be updated independently from the image.

Run the synthetic data generator inside the container the first time:

```bash
docker compose exec api python data/generate_synthetic_data.py --records 500
```

Then run ingestion to build the vector store:

```bash
docker compose exec api python ingestion/run_ingestion.py
```

Once the container is healthy, the API is available at `http://127.0.0.1:8000`.

## Evaluation and tests

Run the automated tests:

```bash
pytest -q
```

Run the graph evaluation cases:

```bash
python evaluation/run_eval.py
```

## Production notes

- Set `MAINTENANCE_COPILOT_ALLOW_PUBLIC=false` and configure `MAINTENANCE_COPILOT_API_KEYS`.
- Set `MAINTENANCE_COPILOT_CORS_ORIGINS` to explicit frontend origins. `*` is rejected at startup.
- Keep `/metrics` behind an admin key and scrape it from your monitoring stack.
