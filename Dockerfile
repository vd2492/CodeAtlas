FROM python:3.12-slim-bookworm

ARG GRAPHIFY_VERSION=0.8.39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODEATLAS_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        gh \
        git \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "graphifyy==${GRAPHIFY_VERSION}"

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin codeatlas \
    && mkdir -p /app/data \
    && chown -R codeatlas:codeatlas /app

COPY --chown=codeatlas:codeatlas app ./app
RUN chmod 0755 /app/app/repos/git_askpass.py

USER codeatlas

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
