# Contributing to TokenFlow Router

Thank you for your interest in contributing!

## Local development setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/sauravdev/TokenFlow-Router
cd TokenFlow-Router
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage
pytest tests/ --cov=tokenflow --cov-report=term-missing

# Single file
pytest tests/unit/test_router.py -v
```

## Linting and type checking

```bash
ruff check tokenflow/ tests/     # lint
ruff format tokenflow/ tests/    # format
mypy tokenflow/ --ignore-missing-imports  # type check
```

Pre-commit will run these automatically on each commit.

## Running locally with Docker

```bash
docker-compose up -d
# Register a mock endpoint
curl -X POST http://localhost:8080/admin/endpoints \
  -H "Content-Type: application/json" \
  -d '{"name":"local-nim","nim_url":"http://mock-nim-1:9001","model_name":"meta/llama-3.1-8b-instruct","gpu_name":"L40S","cost_class":"standard"}'
# Test routing
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.1-8b-instruct","messages":[{"role":"user","content":"hello"}]}'
```

## Architecture overview

```
tokenflow/
├── main.py          — FastAPI app factory, lifespan, middleware
├── config.py        — Settings from environment variables
├── models.py        — Core Pydantic models and enums
├── classifier.py    — Request classification (token counting, workload type)
├── router.py        — Scoring engine and decision engine
├── registry.py      — In-memory endpoint registry
├── telemetry.py     — Background telemetry collection + EMA smoothing
├── observability.py — Prometheus metrics, trace store, workload report
├── profiles.py      — Dynamic backend profile templates (lazy activation)
├── policy_engine/   — Policy DSL, budget/RPM tracking
├── gateway/         — OpenAI-compatible inference proxy
├── admin/           — Admin REST API
└── adapters/        — Per-backend telemetry clients (NIM, vLLM, SGLang, Dynamo)
```

## Submitting a pull request

1. Fork the repo and create a branch from `main`
2. Write tests for any new behaviour
3. Ensure `pytest`, `ruff`, and `mypy` all pass
4. Open a PR with a clear description of what changed and why

## Reporting issues

Open a GitHub issue with:
- TokenFlow version
- Python version
- Steps to reproduce
- Expected vs actual behaviour
