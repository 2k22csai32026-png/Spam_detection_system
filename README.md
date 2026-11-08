# Real-Time Spam Classification Engine & Audit Pipeline

A production-grade machine learning microservice for asynchronous text classification and threat detection, built with FastAPI, Scikit-learn, MongoDB Atlas, and Docker.

## System Architecture

- **Classifier Engine**: Scikit-learn pipeline utilizing TF-IDF Vectorization and Multinomial Naive Bayes.
- **API Layer**: FastAPI asynchronous REST endpoints with Pydantic v2 schema validation.
- **Persistence Store**: Cloud-hosted MongoDB Atlas audit trail managed asynchronously via Motor.
- **Containerization**: Multi-stage lightweight Docker image (`python:3.12-slim`).

## API Endpoints

- `POST /predict` — Evaluates payload, returns verdict (`Spam` vs `Ham`), confidence score, and logs entry to MongoDB.
- `GET /logs` — Retrieves recent historical classification audits.
- `GET /docs` — Interactive OpenAPI / Swagger UI documentation.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/spam-detection-engine.git
   cd spam-detection-engine