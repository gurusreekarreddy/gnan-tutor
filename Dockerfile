FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir openenv-core pydantic uv fastapi uvicorn

COPY . /app

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true

CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]