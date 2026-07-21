# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies into an isolated virtual environment so the final image can
# copy only the runtime environment instead of the full build toolchain.
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Multi-stage builds keep the final image smaller because only the prepared
# virtual environment and application code are copied into the runtime stage.
COPY --from=builder /opt/venv /opt/venv

# The code is baked into the image because it changes with deployments, while
# SQLite/Chroma data will be mounted as volumes so it can persist and remain mutable.
COPY api ./api
COPY graph ./graph
COPY ingestion ./ingestion
COPY retrieval ./retrieval
COPY data/generate_synthetic_data.py ./data/generate_synthetic_data.py
COPY requirements.txt ./requirements.txt

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/chroma_data \
    && chown -R appuser:appuser /app

# Running as a non-root user reduces container breakout impact and follows the
# principle of least privilege for production workloads.
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]