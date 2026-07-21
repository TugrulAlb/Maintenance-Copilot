# Maintenance Copilot

Maintenance Copilot is a portfolio project for semantic and analytical question answering over industrial maintenance and fault logs.

## What is here

- `data/generate_synthetic_data.py` creates a SQLite database with realistic maintenance tickets and LLM-generated technician notes.
- `ingestion/`, `retrieval/`, `graph/`, and `api/` are the main application layers that will be filled in next.
- `tests/` is reserved for unit and integration coverage.
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

## Next steps

- Add ingestion pipelines for SQLite records and text chunking.
- Build hybrid retrieval with embeddings plus BM25.
- Expose question answering through FastAPI.