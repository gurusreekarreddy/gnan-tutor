FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir openenv-core pydantic uv

COPY . /app

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true

CMD ["uv", "run", "--project", ".", "server", "--port", "7860"]