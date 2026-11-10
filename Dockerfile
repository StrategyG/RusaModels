FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN groupadd --system --gid 10001 rusa && useradd --system --uid 10001 --gid rusa --home-dir /app rusa
COPY pyproject.toml README.md ./
COPY rusamodels ./rusamodels
COPY services ./services
COPY migrations ./migrations
COPY alembic.ini ./
RUN pip install --no-cache-dir .
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"
CMD ["uvicorn", "services.model_server.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
