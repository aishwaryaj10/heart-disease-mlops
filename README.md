# Heart Disease Prediction — End-to-End MLOps Project

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for binary heart disease classification using the Heart Disease dataset.

It demonstrates data preparation, exploratory analysis, preprocessing, model comparison, experiment tracking, model packaging, automated testing, containerization, Kubernetes deployment, and application monitoring.

**Project objectives**
- Prepare and validate the dataset.
- Train and compare Logistic Regression and Random Forest classifiers.
- Select the final model using cross-validation F1-score.
- Track experiments and evaluation metrics using MLflow.
- Serve predictions through a REST API built with FastAPI.
- Automate testing using pytest and GitHub Actions.
- Containerize the application using Docker.
- Deploy the API to a local Kubernetes cluster.
- Monitor requests, prediction counts, response times, and application logs.

This is an educational machine learning project, not a clinical diagnostic system.

## 2. Technology Stack

- **Language:** Python 3.12
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn
- **Experiment tracking:** MLflow
- **Model persistence:** joblib
- **API:** FastAPI and Uvicorn
- **Testing:** pytest
- **CI/CD:** GitHub Actions
- **Containerization:** Docker
- **Orchestration:** Kubernetes via Docker Desktop
- **Monitoring:** Prometheus client metrics and Python logging

## 3. Project Structure

```text
heart-disease-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   └── main.py
├── artifacts/
│   ├── heart_disease_model.joblib
│   ├── model_comparison.csv
│   └── model_metadata.json
├── data/
│   ├── raw/
│   └── processed/
├── deployment/
│   └── k8s.yaml
├── notebooks/
├── reports/
├── screenshots/
├── src/
│   └── heart_mlops/
│       ├── data.py
│       ├── preprocessing.py
│       ├── predict.py
│       └── train.py
├── tests/
│   ├── test_api.py
│   └── test_predict.py
├── Dockerfile
├── .dockerignore
├── .gitignore
└── requirements.txt
```

## 4. Setup and Installation

### Prerequisites

Install Python 3.12, Git, and Docker Desktop. Kubernetes is required for the Kubernetes deployment steps.

Clone the repository:

```bash
git clone https://github.com/aishwaryaj10/heart-disease-mlops.git
cd heart-disease-mlops
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Model Training and Evaluation

Run the training pipeline from the project root:

```powershell
python -m src.heart_mlops.train
```

The pipeline prepares the dataset, creates a stratified training/test split, evaluates candidate classifiers, performs five-fold cross-validation on the training split, and logs experiment results to MLflow.

The candidate models are:

1. Logistic Regression
2. Random Forest

The final candidate is selected using mean cross-validation F1-score. The held-out test set is used for final evaluation rather than model selection.

The pipeline saves the selected model, model comparison results, and model metadata in the `artifacts/` directory.

### Model Evaluation

The recorded evaluation results were:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Test accuracy | 0.8333 | 0.8667 |
| Test precision | 0.8462 | 0.8846 |
| Test recall | 0.7857 | 0.8214 |
| Test F1-score | 0.8148 | 0.8519 |
| Test ROC-AUC | 0.9487 | 0.9403 |
| Mean cross-validation F1 | 0.8035 | 0.7847 |

**Selected model:** Logistic Regression, based on the higher mean cross-validation F1-score.

These are results from the current experimental run and may vary if the dataset, dependencies, or training configuration changes.

## 6. MLflow Experiment Tracking

Training experiments are logged under the experiment name:

`heart-disease-classification`

The training code configures a local SQLite tracking database named `mlflow.db`.

To launch the MLflow UI from the project root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the local MLflow UI at:

http://127.0.0.1:5000

The UI can be used to compare recorded runs, parameters, and evaluation metrics.

## 7. Automated Testing

Run the complete test suite:

```powershell
python -m pytest tests -v
```

The current test suite contains nine tests covering the API endpoints, prediction output, required features, and input validation.

**Verified result:** 9 tests passed.

A Starlette/httpx deprecation warning was present in the local test run but did not cause test failures.

## 8. Run the API Locally

Start FastAPI from the project root:

```powershell
uvicorn api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

Prometheus metrics:

http://127.0.0.1:8000/metrics

### Prediction Endpoint

The API provides:

`POST /predict`

Example request:

```json
{
  "age": 52,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 3
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.1328
}
```

The values above are an example response from a verified run, not a guarantee for every execution. The prediction represents the model's predicted class, and the probability is the estimated probability for class 1.

The model output must not be used as a substitute for professional medical assessment.

## 9. Docker Deployment

Build the Docker image:

```powershell
docker build -t heart-disease-api:1.2 .
```

Start the container:

```powershell
docker run -d --name heart-disease-container -p 8000:8000 heart-disease-api:1.2
```

If a container with the same name already exists, remove that container before running the command again:

```powershell
docker rm -f heart-disease-container
```

Check the running container:

```powershell
docker ps
```

View application logs:

```powershell
docker logs --tail 30 heart-disease-container
```

The container exposes port 8000 and runs the FastAPI application using Uvicorn.

## 10. Kubernetes Deployment

The Kubernetes configuration is available at `deployment/k8s.yaml`.

Ensure Docker Desktop Kubernetes is enabled and the local Docker image is available to the cluster.

Apply the configuration:

```powershell
kubectl apply -f deployment/k8s.yaml
```

Check the deployment:

```powershell
kubectl get nodes
kubectl get deployments
kubectl get pods
kubectl get services
```

The deployment uses one replica and defines readiness and liveness probes against `/health`. The service is configured as a NodePort service on port 30080.

For reliable local access, use port forwarding:

```powershell
kubectl port-forward service/heart-disease-api-service 8081:8000
```

Keep the port-forwarding terminal open and visit:

- Health: http://127.0.0.1:8081/health
- API documentation: http://127.0.0.1:8081/docs
- Metrics: http://127.0.0.1:8081/metrics

**Deployment scope:** This configuration has been verified on a local Docker Desktop Kubernetes cluster. It does not represent a public cloud deployment.

## 11. Monitoring and Logging

The FastAPI application exposes Prometheus-compatible metrics at `/metrics`.

Custom metrics include:

- `http_requests_total`: request counts by HTTP method, endpoint, and status code.
- `http_request_duration_seconds`: request processing duration.
- `heart_disease_predictions_total`: successful predictions grouped by predicted class.

The application logs HTTP request methods, paths, status codes, durations, and unexpected request errors. The logs are written to standard output and can be collected through Docker.

Example:

```powershell
docker logs --tail 30 heart-disease-container
```

The metrics endpoint provides instrumentation for monitoring. A separate Prometheus server and dashboard are not configured by this project.

## 12. Continuous Integration

GitHub Actions is configured in `.github/workflows/ci.yml`.

On pushes and pull requests targeting the configured main branches, the workflow checks out the repository, sets up Python 3.12, installs dependencies, and runs:

```bash
python -m pytest tests -v
```

The workflow was verified as successful after the latest project changes.

## 13. Reproducibility Notes

- Run commands from the repository root unless otherwise stated.
- Use the supplied requirements file to install dependencies.
- Use the training command to regenerate model artifacts.
- The saved model and metadata are included in the repository for API execution.
- MLflow's local database and experiment outputs are generated locally.
- Docker and Kubernetes deployment commands assume the required local tools are installed and running.

## 14. Limitations

- The model is intended for educational demonstration only.
- Evaluation metrics describe one experimental run and do not establish clinical validity.
- The project is deployed locally, not to a public cloud.
- The metrics endpoint exposes instrumentation but does not include a separately deployed monitoring dashboard.
- Model predictions may be affected by dataset limitations, population differences, and data quality.

## 15. Repository

GitHub: https://github.com/aishwaryaj10/heart-disease-mlops
