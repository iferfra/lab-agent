FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --group frontend --no-dev --no-install-project --no-install-workspace

COPY ./frontend ./frontend

EXPOSE 10000

CMD ["uv", "run", "chainlit", "run", "frontend/app.py", "--host", "localhost", "--port", "10000"]
