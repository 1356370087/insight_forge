FROM python:3.12-slim AS runtime
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 2024
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:2024/healthz', timeout=3)"
CMD ["uvicorn", "open_deep_research.server:app", "--host", "0.0.0.0", "--port", "2024", "--timeout-graceful-shutdown", "30"]
