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

## Routing architecture

The LangGraph workflow starts with an `input_guardrail` node. If the question is
clearly off-topic or looks like prompt injection, the graph routes directly to
`compose` with a safe refusal. Otherwise it continues to a structured LLM router.
The router returns an `intent` (`analytical`, `semantic`, or `hybrid`),
confidence, a short reasoning string, and safe metadata filters such as
`production_line`, `fault_category`, `machine_id`, and `severity`.

- `analytical` questions go to the NL2SQL agent over `maintenance_logs`.
- `semantic` questions go to hybrid retrieval: dense vector search plus BM25,
  Reciprocal Rank Fusion, then reranking.
- `hybrid` questions run both NL2SQL and hybrid retrieval before answer
  generation.

After answer generation, the graph runs an evidence-aware `evaluate_answer`
reflection step. The evaluator checks whether the draft addresses the question,
is supported by the evidence, and includes the needed citations. If it finds
gaps, it can route either to `answer` for a composition-only retry or back to
`analytical`/`semantic` for one evidence re-acquisition attempt. Hybrid questions
can return plural `retry_targets` because SQL and retrieval evidence may
independently be insufficient. Answer retries and evidence retries use separate
counters because re-running SQL/retrieval is more expensive than rewriting an
answer. A hard cap prevents infinite loops; when a cap is reached, the compose
node returns the best available answer with a soft caveat.

When no model credentials are configured, the router falls back to a deterministic
local classifier with the same output shape so tests and local demos still run.

## Guardrails

The guardrails are intentionally lightweight and implemented in `graph/guardrails.py`
instead of using a heavy external framework. The structure still follows the same
pattern used by production guardrail systems: input rails protect the graph before
work starts, and output rails protect the response before it leaves the API.

Input guardrails:

- Off-topic detection blocks questions outside industrial maintenance, such as
  creative writing or general trivia requests.
- Prompt injection detection blocks attempts like "ignore previous instructions"
  or requests to reveal hidden prompts.
- Blocking happens in the first LangGraph node. This saves cost because an
  invalid request does not run classification, retrieval, SQL generation, answer
  generation, or evaluation.

Output guardrails:

- Regex-based PII redaction removes obvious emails, phone numbers, and personal
  ID-like values if they accidentally appear in the final answer.
- Empty or low-confidence answers are converted into a clear fallback such as
  "I do not have enough information to answer confidently" instead of returning
  a made-up-sounding response.
- The compose path always returns the API shape expected by `AnswerResponse`,
  including refusal/error paths.

In a larger production system this maps cleanly to frameworks such as NeMo
Guardrails or Guardrails AI. Those tools formalize the same input/output rail
concept with more configuration, policies, and runtime integrations.

## Synthetic data generation

The data generator uses an OpenAI-compatible client to batch-generate natural language fault descriptions so later semantic search tests have real linguistic variety to work with. Structured fields are generated locally to keep the dataset deterministic and easy to inspect.

Run it after setting environment variables from `.env.example`:

```bash
python data/generate_synthetic_data.py --records 500
```

For Azure OpenAI endpoints that use the OpenAI-compatible path, configure:

```bash
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://topuz-openai.openai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4-mini
```

Use `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` separately for ingestion because
retrieval embeddings require an embedding-capable deployment.

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

Run the RAGAS evaluation pipeline:

```bash
python eval/run_ragas_eval.py
```

The RAGAS runner executes the full LangGraph workflow for curated analytical,
semantic, and hybrid questions, then saves a per-question table to
`eval/results.csv`.

Before running it, generate data and build retrieval indexes:

```bash
python data/generate_synthetic_data.py --records 500
python ingestion/run_ingestion.py
```

The runner can build the BM25 index from SQLite if it is missing, but ChromaDB
requires ingestion because embeddings must be generated first.

Metrics:

- `faithfulness`: checks whether the final answer is supported by the retrieved
  context or SQL evidence. This is reference-free: it only needs the question,
  answer, and evidence.
- `answer_relevancy`: checks whether the answer actually addresses the user's
  question. This is also reference-free.
- `context_precision`: checks whether retrieved contexts are relevant and ranked
  usefully for semantic/hybrid questions.
- `context_recall`: checks whether the retriever found the contexts that should
  have been found. This needs a ground-truth reference context, usually curated
  record ids, because recall requires knowing what the system was supposed to
  retrieve.
- `hallucination_flag_rate`: custom LLM-as-judge metric that reports the share
  of answers containing claims not present in the evidence. It is simpler than
  RAGAS faithfulness and easier to explain in an interview.

Placeholder result table:

| Run date | Faithfulness | Answer relevancy | Context precision | Context recall | Hallucination flag rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| TODO | TODO | TODO | TODO | TODO | TODO |

## Production notes

- Set `MAINTENANCE_COPILOT_ALLOW_PUBLIC=false` and configure `MAINTENANCE_COPILOT_API_KEYS`.
- Set `MAINTENANCE_COPILOT_CORS_ORIGINS` to explicit frontend origins. `*` is rejected at startup.
- Keep `/metrics` behind an admin key and scrape it from your monitoring stack.
