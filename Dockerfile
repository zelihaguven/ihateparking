FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3000 \
    IHATEPARKING_OUTPUT_DIR=/tmp/ihateparking-runtime \
    IHATEPARKING_MODEL_CHECKPOINT=/app/deployment/export/perception_model_numpy_mlp_v2.npz

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ihateparking ./ihateparking
COPY scripts ./scripts
COPY web ./web
COPY assets/*.mp4 ./assets/
COPY assets/*.svg ./assets/
COPY artifacts/trace.json artifacts/footage_assist.json ./artifacts/
COPY deployment/export ./deployment/export
COPY manus-routes.json ./manus-routes.json

RUN python -m pip install --upgrade pip \
    && python -m pip install '.[dashboard,vision,mpc,mpc-ad,perception]' \
    && useradd --create-home --uid 10001 ihateparking \
    && mkdir -p /tmp/ihateparking-runtime \
    && chown -R ihateparking:ihateparking /app /tmp/ihateparking-runtime

USER ihateparking
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','3000')+'/api/health',timeout=3)"]

ENTRYPOINT ["python", "-m", "ihateparking.dashboard"]
CMD ["--host", "0.0.0.0"]
