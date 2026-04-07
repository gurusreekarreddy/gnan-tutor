FROM python:3.10-slim

WORKDIR /app

RUN pip install openenv-core pydantic > /dev/null 2>&1

COPY . /app

EXPOSE 8000

ENV ENABLE_WEB_INTERFACE=true
CMD ["openenv", "start"]
