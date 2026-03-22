FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8080

ENV TOKENFLOW_HOST=0.0.0.0
ENV TOKENFLOW_PORT=8080

CMD ["uvicorn", "tokenflow.main:app", "--host", "0.0.0.0", "--port", "8080"]
