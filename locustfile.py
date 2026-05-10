from locust import HttpUser, task, between
import json

class HeartDiseaseUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        payload = {
            "age": 63,
            "trestbps": 145,
            "chol": 233,
            "thalch": 150,
            "oldpeak": 2.3,
            "ca": 0,
            "sex": "Male",
            "cp": "typical angina",
            "fbs": "TRUE",
            "restecg": "lv hypertrophy",
            "exang": "FALSE",
            "slope": "downsloping",
            "thal": "fixed defect"
        }
        self.client.post("/predict", json=payload)

    @task
    def health(self):
        self.client.get("/health")