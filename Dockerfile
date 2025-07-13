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
COPY .env /app/skincare
WORKDIR /app/skincare

EXPOSE 7860

ENTRYPOINT ["python", "main.py"]