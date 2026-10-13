FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8010 \
    IHATEPARKING_OUTPUT_DIR=/tmp/ihateparking-runtime

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ihateparking ./ihateparking
COPY web ./web
COPY deployment/export ./deployment/export

RUN python -m pip install --upgrade pip \
    && python -m pip install .[dashboard,vision] \
    && useradd --create-home --uid 10001 ihateparking \
    && mkdir -p /tmp/ihateparking-runtime \
    && chown -R ihateparking:ihateparking /app /tmp/ihateparking-runtime

USER ihateparking
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/health', timeout=3)"

ENTRYPOINT ["python", "-m", "ihateparking.dashboard"]
CMD ["--host", "0.0.0.0", "--port", "8010"]
