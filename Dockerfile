FROM public.ecr.aws/docker/library/debian:trixie
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/bin/
RUN useradd -u 1000 -d /app app \
    && apt-get update \
    && apt-get install -y ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*


ENV UV_CACHE_DIR=/uv-cache/
ENV UV_PYTHON_INSTALL_DIR=/opt/python/
ENV UV_LINK_MODE=copy
WORKDIR /app
RUN --mount=type=cache,target=/uv-cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/uv-cache \
	uv sync --locked --compile-bytecode --no-dev --group fastapi

USER 1000
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["fastapi", "run", "-e", "horde_openai_proxy.proxy:app"]
