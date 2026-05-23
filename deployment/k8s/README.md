# Kubernetes Deployment

These manifests support a local Minikube or Docker Desktop Kubernetes submission.

## Minikube quick run

```bash
minikube start
eval $(minikube docker-env)
docker build -t bits-mlops-heart:latest .
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl -n bits-mlops rollout status deployment/heart-disease-api
kubectl -n bits-mlops get pods,svc
```

Use the NodePort service:

```bash
minikube service heart-disease-api -n bits-mlops --url
```

Or use port forwarding:

```bash
kubectl -n bits-mlops port-forward svc/heart-disease-api 8000:80
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input/patient_high_risk.json
```

## Monitoring

The API exposes Prometheus metrics at `/metrics`. If the Prometheus Operator is
installed, apply `prometheus-service-monitor.yaml`; otherwise configure
Prometheus to scrape the service endpoint directly.
