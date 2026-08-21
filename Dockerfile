FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISCLOSURE_DATA_DIR=/app/data/corpus

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/corpus ./data/corpus

# The corpus check stays available as `docker run <image> python -m app`.
EXPOSE 8000
CMD ["python", "-m", "app.api"]
