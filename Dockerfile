FROM python:3.10-slim

WORKDIR /app

RUN pip install openenv-core pydantic > /dev/null 2>&1

COPY . /app

EXPOSE 7860

ENV ENABLE_WEB_INTERFACE=true
CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]