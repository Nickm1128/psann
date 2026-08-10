FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel \
      --no-cache-dir \
      --no-deps \
      --retries 5 \
      --timeout 60 \
      --wheel-dir /wheels \
      .

FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PSANN_ARTIFACT_PATH=/artifacts/model.psann

WORKDIR /opt/psann

RUN apt-get -o Acquire::Retries=5 update --error-on=any \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/*

COPY constraints/deployment-py311.txt /tmp/deployment-py311.txt
RUN python -m pip install --no-cache-dir \
      --retries 5 \
      --timeout 60 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0
RUN python -m pip install \
      --no-cache-dir \
      --retries 5 \
      --timeout 60 \
      -r /tmp/deployment-py311.txt
COPY --from=builder /wheels/psann-*.whl /tmp/
RUN python -m pip install --no-cache-dir --no-deps /tmp/psann-*.whl \
    && rm /tmp/psann-*.whl \
    && python -m pip check

RUN addgroup --system psann && adduser --system --ingroup psann psann
USER psann

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"

CMD ["python", "-m", "psann.serving", "--host", "0.0.0.0", "--port", "8080"]
