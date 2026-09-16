FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ./

RUN uv sync --locked

COPY "predict.py" "model.bin" ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
