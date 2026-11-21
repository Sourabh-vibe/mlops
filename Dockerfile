# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS wheels
WORKDIR /build
COPY requirements-serving.lock .
RUN python -m pip wheel --no-cache-dir --require-hashes --only-binary=:all: \
    --wheel-dir /wheels -r requirements-serving.lock

FROM python:3.13-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IRIS_MODEL_DIR=/app/models/iris \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements-serving.lock /tmp/requirements-serving.lock
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    python -m pip install --no-cache-dir --no-index --require-hashes \
      --only-binary=:all: --find-links=/wheels -r /tmp/requirements-serving.lock \
    && groupadd --gid 10001 iris \
    && useradd --uid 10001 --gid iris --no-create-home iris
COPY src/__init__.py src/schema.py src/artifacts.py ./src/
COPY api/ ./api/
# Build only after `dvc repro` or `dvc pull` has materialized the approved model.
COPY models/iris/model.joblib models/iris/metadata.json ./models/iris/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
