FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1000 vscode && useradd --uid 1000 --gid vscode --create-home --shell /bin/bash vscode
COPY pyproject.toml README.md ./
COPY rusamodels ./rusamodels
COPY services ./services
COPY migrations ./migrations
COPY alembic.ini ./
RUN python -m pip install --no-cache-dir --upgrade "pip>=25.3" "setuptools>=84.1.0" "wheel>=0.46.2" \
    && python -m pip install --no-cache-dir . \
    && python -m pip check \
    && find /usr/local/lib/python3.12/site-packages -maxdepth 1 -type d \
        \( -name 'setuptools-70.3.0.dist-info' -o -name 'setuptools-70.3.0.egg-info' \) \
        -prune -exec rm -rf {} + \
    && chown -R vscode:vscode /app
ENV HOME=/home/vscode PATH=/home/vscode/.local/bin:$PATH
USER vscode
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["uvicorn", "services.model_server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
