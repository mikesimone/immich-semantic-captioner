ARG CUDA_BASE_IMAGE
FROM ${CUDA_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /app/requirements.txt

# BuildKit pip cache (fast rebuilds when requirements change)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# captioner.py is bind-mounted from docker-compose.yml:
#   ./captioner/captioner.py:/app/captioner.py:ro

CMD ["/opt/venv/bin/python", "/app/captioner.py"]
