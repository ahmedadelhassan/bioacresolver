FROM python:3.8

ENV PYTHONUNBUFFERED 1

EXPOSE 8080:8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . ./
ENV PYTHONPATH=/app/src
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT ./docker-entrypoint.sh
