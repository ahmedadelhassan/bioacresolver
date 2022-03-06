FROM python:3.8

ENV PYTHONUNBUFFERED 1

EXPOSE 8080:8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . ./
ENV PYTHONPATH=/app/src
CMD [ "uvicorn", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info", "--factory", "apis:create_app" ]
