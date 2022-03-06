# Bioacresolver

> A simple service to train and deploy an ML model
> for acronym disambiguation NLP task on Dutch medical data

# 🚀 Setup guide

1. Configure `.env` file for your choice. You can put there anything you like, it will be used to configure you
   services (optional)
2. Configure `settings.toml` file with model parameters (optional)
3. Run `docker compose up`
4. Access the different service
    - Prediction API docs: http://localhost:5000/docs
    - MLflow server: http://localhost:5000
    - S3 artifact store browser: http://localhost:9001/ (credentials in .env file)

---

## Service Details

### Data storage & pipeline

- Data processing using pandas
- Storage partially on S3-like local service using [Min.io](https://min.io/)

### ML model

- Transformer architecture trained for sequence-to-sequence task
- Model pipeline consists of the following layers:
    - TextVectorization: embeds input/output sequences as numerical vectors
    - TransformerEncoder: encodes input sequences using self multiheaded attention
    - PositionalEmbedding: encodes the positions of the sequences
    - TransformerDecoder: decodes sequences using multiheaded attention on both the input and output sequences
- Hyperparameter tuning was performed only manually, and automated tuning should be implemented
- The model was only trained on toy dataset, therefore the vocabulary has low coverage, and the accuracy is low for 
  out-of-sample evaluation

### Model tracking

- MLflow is used to track experiments and training runs
- Logging model artifacts using keras autologging

### Model serving

- Deployed as a service with a prediction API using FastAPI

## Licensing

The service adapts the dockerized MLflow project from [mlflow-docker](https://github.com/Toumash/mlflow-docker)
