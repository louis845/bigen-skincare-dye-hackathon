FROM python:3.10-slim AS base

RUN pip install poetry
RUN mkdir /install
COPY pyproject.toml /install
COPY poetry.lock /install

WORKDIR /install
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

RUN mkdir /app
WORKDIR /app

COPY src/skincare /app/skincare
COPY .env /app
COPY main.py /app

EXPOSE 8000

ENTRYPOINT ["python", "main.py"]