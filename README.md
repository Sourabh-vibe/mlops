<<<<<<< Updated upstream
# mlops
=======
# Iris Flower Classification MLOps

A complete local training and serving pipeline for scikit-learn's Iris dataset.
Random Forest classifies setosa, versicolor, and virginica from four measurements
in centimeters. Data and model bytes live in DVC; experiment history and the
scikit-learn model are also logged in MLflow. The serving image contains the
tested model and inference dependencies.

## Project layout

```text
.
├── .dvc/                       # DVC configuration; cache stays out of Git
├── .github/workflows/ci.yml     # Lint, DVC, tests, image build, HTTP smoke test
├── api/
│   ├── __init__.py
│   └── app.py                  # /predict, /healthz, /readyz
├── data/raw/iris.csv            # Generated; DVC cached
├── models/iris/                 # Generated; DVC cached as one bundle
│   ├── model.joblib
│   └── metadata.json
├── reports/
│   ├── data_validation.json    # Git-friendly validation and dataset checksum
│   └── metrics.json            # Git-friendly DVC metrics
├── src/
│   ├── __init__.py
│   ├── schema.py               # Shared feature order, labels, checksums, JSON IO
│   ├── config.py               # Strict parameter validation
│   ├── data.py                 # Export and validation commands
│   ├── artifacts.py            # Bundle export and compatibility checks
│   └── train.py                # Stratified split, training, evaluation, MLflow
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── dvc.yaml
├── dvc.lock                    # Generated lineage; commit this file
├── params.yaml
├── pyproject.toml
├── requirements.txt            # Full development/training/test environment
├── requirements.lock           # Hashed transitive training/test dependency lock
├── requirements-serving.txt    # Smaller serving environment
└── requirements-serving.lock   # Hashed transitive serving dependency lock
```

## Run locally

Use Python 3.13; CI, Docker, and the dependency locks target 3.13. Run from the repository
root. The dataset ships with scikit-learn, so training does not download data.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements.lock
python -m pip check
```

If this is a fresh copy without Git/DVC metadata, initialize once:

```bash
git init
dvc init
```

Then run the entire pipeline and test suite:

```bash
dvc repro
pytest --cov --cov-report=term-missing
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> to launch the interactive API docs (also available
directly at <http://127.0.0.1:8000/docs>). Training must finish
before starting the service. It never trains as a side effect of import or an
HTTP request. Missing, corrupted, or incompatible bundles fail service startup.

The stages can also be invoked individually:

```bash
python -m src.data prepare --output data/raw/iris.csv
python -m src.data validate --data data/raw/iris.csv --report reports/data_validation.json
python -m src.train --data data/raw/iris.csv --params params.yaml --output models/iris --metrics reports/metrics.json
```

## Reproducibility and quality

Validation requires 150 rows, 50 integer targets per class, the exact feature
schema, and positive, finite numeric measurements. A legitimate duplicate in
Iris is preserved. Train/test separation uses a seeded, stratified 80/20 split
(120 training rows and 30 test rows). The holdout is never used to fit the model.
The default forest has 200 trees, depth 5, and random state 42.

Accuracy, macro precision, macro recall, and macro F1 are logged against the held
out rows. Training fails below `quality.min_accuracy` (0.90), and does not replace
the previously exported bundle or metrics on that failure. A failed MLflow run
retains its metrics for diagnosis. The exported estimator is the evaluated
120-row model; it is not silently refitted on the test set. This small benchmark
holdout provides limited evidence of performance on new real-world data.

Change `params.yaml` and run `dvc repro` to retrain the affected stages. DVC tracks
source files, dependencies, parameters, data hashes, and model outputs in
`dvc.lock`. Reports remain human-readable Git files. Runtime metadata records
the dataset SHA-256, model SHA-256/version, parameters, scikit-learn version,
MLflow run ID, and MLflow model URI.

Top-level requirements and transitive dependencies are pinned; the checked-in
lockfiles include hashes and platform markers. CI and Docker install the locks
with `--require-hashes`. Update the input `.txt` files and regenerate both locks
before validating dependency changes (the serving lock is constrained to the
tested training versions):

```bash
uv pip compile requirements.txt --python-version 3.13 --universal --generate-hashes -o requirements.lock
uv pip compile requirements-serving.txt -c requirements.lock --python-version 3.13 --universal --generate-hashes -o requirements-serving.lock
```

The Docker base uses the maintained
`python:3.13-slim-bookworm` tag; pin its reviewed digest if your release policy
requires an immutable base image. Rebuild and validate updates through CI.

## Git and DVC storage

Commit source/configuration, `.dvc/config`, `.dvc/.gitignore`, `dvc.yaml`,
`dvc.lock`, and reports to Git. Dataset/model bytes are ignored by Git and cached
by DVC. They are pipeline outputs, so do **not** also run `dvc add` on them.
The checked-in lock file records hashes; materialize its bytes using `dvc pull`
when a shared remote exists, or `dvc repro` to regenerate locally.

Configure an organization-controlled remote before sharing large artifacts.
For example, a mounted storage directory requires no extra DVC storage plugin:

```bash
dvc remote add -d artifacts /absolute/path/to/shared/iris-dvc-store
dvc push
git add .
git commit -m "Add Iris MLOps pipeline and artifact lineage"
# On another checkout with access to that same remote:
dvc pull
```

For S3, install the compatible `dvc[s3]==3.64.0` extra and use
`dvc remote add -d artifacts s3://YOUR-BUCKET/iris`. Use environment credentials
or `dvc remote modify --local`; never commit access keys. No shared remote or
cloud credentials are assumed by this project. DVC restoration does not restore
MLflow's database/history; back up its database and artifact store separately.

## MLflow

The default tracking backend is `sqlite:///mlflow.db` in the working directory,
with local MLflow artifacts. To inspect the same backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>. Each run logs forest parameters, split and quality
settings, metrics, dataset checksum, confusion matrix, classification report,
and a scikit-learn MLflow model with a signature and input example. CI also tags
runs with `GITHUB_SHA`. For a shared server, export `MLFLOW_TRACKING_URI` before
training; e.g. `export MLFLOW_TRACKING_URI=https://mlflow.example.com` on Bash or
`$env:MLFLOW_TRACKING_URI='https://mlflow.example.com'` on PowerShell. Supply that
server's authentication through its supported environment configuration.

MLflow connectivity is required during training. Inference uses the local
bundle and is independent of the tracking server. DVC can restore a cached run
without creating a new MLflow experiment; use `dvc repro --force` when a fresh
tracking run is required. CI deliberately forces a fresh validation/training run.

## Prediction contract

All four fields are required, positive, finite JSON numbers in centimeters.
Extra fields, strings, booleans, missing values, and invalid measurements receive
HTTP 422. Numbers exceeding the estimator's supported numeric range also receive
HTTP 422, including overflowed JSON numbers. Key order is irrelevant.

```bash
curl http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Example response (version and scores depend on the trained artifact):

```json
{
  "species": "setosa",
  "class_id": 0,
  "confidence": 1.0,
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0},
  "model_version": "0123456789ab"
}
```

`confidence` is the maximum Random Forest `predict_proba` score. These scores
sum to one but are not calibrated guarantees. Class IDs 0/1/2 map to
setosa/versicolor/virginica. `/healthz` reports process liveness; `/readyz` reports
model availability and version. Model-unavailable and inference-failure responses
use HTTP 503 without exposing internal error details. A bundle is loaded once
per process. Deploy a new immutable image to change models.

## Container

Materialize a passing bundle before building. The multi-stage build creates
dependency wheels and installs only serving dependencies in the final image.
It copies the tested artifact without retraining, runs as UID 10001, and probes
`/readyz` with the Python standard library.

```bash
dvc repro
docker build -t iris-mlops:local .
docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges -p 127.0.0.1:8000:8000 iris-mlops:local
```

Only load trusted joblib bundles: deserialization executes pickle code. SHA-256
detects corruption, while artifact provenance and access control establish trust.
Startup verifies the checksum, feature order, class mapping, fitted model type,
and exact scikit-learn version. Keep training and serving dependencies aligned.

## CI and delivery

GitHub Actions runs on pull requests, pushes to main/master, and manual dispatch.
It installs dependencies, checks formatting/lint, executes all DVC stages, runs
unit and integration tests with an 85% coverage gate, and uploads the tested model,
reports, and lockfile. A dependent job builds that exact bundle into an image and
tests readiness and an HTTP prediction in a non-root, read-only container.

After container tests pass, pushes to main/master publish that tested image to
`ghcr.io/<owner>/<repository>:<git-sha>` using the job-scoped `GITHUB_TOKEN` with
package write permission. Pull requests build/test without publishing. Manual
runs publish only when `publish_image` is selected and the branch is main/master.
Allow Actions to publish the repository's GHCR package in your GitHub settings.
The commit tag identifies the release; use the registry digest when deploying.
No credentials are stored in source.

Production rollout needs your deployment target and release policy. Put public
deployments behind a TLS/auth gateway;
set resource/concurrency limits and collect request/error/latency and model-input
monitoring appropriate to the deployment. No external deployment has been made.

## Local verification

The pipeline was run successfully, and a repeated `dvc repro` skipped all stages;
`dvc status --json` returned `{}`. All 51 tests passed with 89.46% branch-aware
coverage. Ruff lint/format and `pip check` passed. The default holdout metrics
were accuracy 0.90, macro precision 0.9024, macro recall 0.90, and macro F1 0.8997.
The local Docker engine was unavailable, so the container build/smoke test and
GitHub workflow execution remain to be verified in their target environments.

## References

- [DVC pipeline definitions and caching](https://dvc.org/doc/user-guide/pipelines/defining-pipelines)
- [MLflow scikit-learn model logging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
>>>>>>> Stashed changes
