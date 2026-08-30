# Disposable BGE-M3 evaluation image. Never used by production; it exists only to
# give the evaluation a real BGE-M3 on the host GPU, isolated from the Festival
# runtime so no large ML package is installed there.
#
# CUDA 12.1 matches the torch cu121 wheels below. The host driver (577.00) is
# newer, which CUDA supports via minor-version compatibility.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch first, from the CUDA index, so FlagEmbedding does not pull a CPU build.
RUN pip install --no-cache-dir \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# The repository pins the embedding dependency; keep the same constraint here so
# the container and the documented contract cannot drift.
COPY requirements-embedding.txt /tmp/requirements-embedding.txt
RUN pip install --no-cache-dir -r /tmp/requirements-embedding.txt psycopg[binary]

WORKDIR /app
