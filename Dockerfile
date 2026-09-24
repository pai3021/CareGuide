# CareGuide Docker 多阶段构建
# 默认镜像保持精简，开发镜像包含调试工具

# ── 阶段 1：基础环境 ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# curl 用于健康检查；不再需要 gcc/g++（已移除本地 ML 模型）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 阶段 2：安装 Python 依赖 ──────────────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt requirements-semantic.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-semantic.txt

# ── 阶段 3：默认运行镜像 ──────────────────────────────────────────────────────
FROM base AS production

# 非 root 用户运行。先创建用户，后续 COPY 直接带 owner，避免 chown -R 复制出额外大层。
RUN useradd -m -u 1000 careguide

# 从依赖阶段复制已安装的包
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=careguide:careguide . .

# 创建必要目录，只调整运行期需要写入的目录权限，避免递归 chown 整个应用。
RUN mkdir -p /app/data/chroma /app/data/runtime /app/data/traces /app/logs /app/config && \
    chown careguide:careguide /app/data /app/data/chroma /app/data/runtime /app/data/traces /app/logs /app/config
USER careguide

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "careguide.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ── 阶段 4：开发镜像 ──────────────────────────────────────────────────────────
FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    chmod -R 777 /app/data /app/logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "careguide.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ── 阶段 5：可选 CPU Transformer 运行镜像 ────────────────────────────────────
# RoBERTa 只负责 13 类意图推理。本机没有 GPU 时使用 PyTorch CPU wheel；
# checkpoint 通过只读 volume 挂载，不复制进镜像或 Git。
FROM dependencies AS transformer-dependencies

COPY requirements-transformer-runtime.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
    pip install -r requirements-transformer-runtime.txt

FROM base AS transformer

RUN useradd -m -u 1000 careguide

COPY --from=transformer-dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=transformer-dependencies /usr/local/bin /usr/local/bin
COPY --chown=careguide:careguide . .

RUN mkdir -p /app/data/chroma /app/data/runtime /app/data/traces /app/logs /app/config /models && \
    chown careguide:careguide /app/data /app/data/chroma /app/data/runtime /app/data/traces /app/logs /app/config
USER careguide

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "careguide.api:app", "--host", "0.0.0.0", "--port", "8000"]
