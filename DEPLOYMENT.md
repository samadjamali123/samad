# Plant Leaf Disease Detection System - Deployment Guide

This comprehensive deployment guide covers production setup for the Plant Leaf Disease Detection System using Docker and Kubernetes.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Configuration](#environment-configuration)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Considerations](#security-considerations)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

**Hardware:**
- CPU: 4 cores (8+ recommended)
- RAM: 8GB (16GB+ recommended)
- Storage: 50GB SSD (100GB+ recommended)
- GPU: Optional, NVIDIA GPU with CUDA support for model acceleration

**Software:**
- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Python 3.9+ (for local development)
- Node.js 16+ (for frontend development)

### Production Requirements

**Hardware:**
- CPU: 8+ cores (16+ recommended)
- RAM: 32GB+ (64GB+ recommended)
- Storage: 200GB+ SSD with high IOPS
- GPU: NVIDIA GPU (RTX 3090/4090 or A100)
- Network: 1Gbps+ bandwidth

**Software:**
- Docker with orchestration (Swarm/Kubernetes)
- Load balancer (NGINX/HAProxy)
- Database cluster (PostgreSQL 14+)
- Redis cluster (for caching)
- Monitoring stack (Prometheus + Grafana)

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Database Configuration
POSTGRES_URL=postgresql://username:password@localhost:5432/plantdetection
POSTGRES_POOL_SIZE=20
POSTGRES_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# AI Model Configuration
GROK_API_KEY=your_grok_api_key_here
PLANTNET_API_KEY=your_plantnet_api_key_here
INATURALIST_API_KEY=your_inaturalist_api_key_here

# File Upload Configuration
MAX_FILE_SIZE_MB=20
MIN_IMAGE_SIZE=100
MAX_CONCURRENT_UPLOADS=10

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=100

# Security
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=https://yourdomain.com

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO

# Feature Flags
ENABLE_ADVANCED_ANALYSIS=true
ENABLE_TREATMENT_CALCULATOR=true
ENABLE_RESULT_SHARING=true
ENABLE_BATCH_ANALYSIS=true
```

### Configuration Files

**nginx/nginx.conf:**
```nginx
upstream backend {
    server backend:8000;
}

upstream frontend {
    server frontend:8501;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # File upload limits
        client_max_body_size 50M;
        proxy_request_buffering off;
    }

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Docker Deployment

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/yourorg/plant-leaf-detection.git
cd plant-leaf-detection
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start services:**
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

4. **Initialize database:**
```bash
# Run database migrations
docker-compose exec backend python -m alembic upgrade head

# Load initial data
docker-compose exec backend python scripts/load_initial_data.py
```

### Production Docker Deployment

1. **Production docker-compose.yml:**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: plantdetection
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - backend
    restart: always

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend
    restart: always

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - ENVIRONMENT=production
      - POSTGRES_URL=${POSTGRES_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - backend
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    environment:
      - API_BASE_URL=${FRONTEND_API_URL}
    networks:
      - backend
    restart: always
    deploy:
      replicas: 2

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    networks:
      - backend
    restart: always

volumes:
  postgres_data:
  redis_data:

networks:
  backend:
    driver: bridge
```

2. **Deploy with SSL:**
```bash
# Place SSL certificates in nginx/ssl/
docker-compose -f docker-compose.prod.yml up -d
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (optional)
- Ingress controller (NGINX/Traefik)
- Persistent storage provisioner

### Kubernetes Manifests

**1. Namespace:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: plantdetection
  labels:
    name: plantdetection
```

**2. ConfigMap:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: plantdetection-config
  namespace: plantdetection
data:
  ENVIRONMENT: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
```

**3. Secrets:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: plantdetection-secrets
  namespace: plantdetection
type: Opaque
data:
  POSTGRES_URL: <base64-encoded>
  REDIS_URL: <base64-encoded>
  GROK_API_KEY: <base64-encoded>
  SECRET_KEY: <base64-encoded>
```

**4. PostgreSQL Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: plantdetection
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: plantdetection
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: plantdetection-secrets
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: plantdetection-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: plantdetection
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: plantdetection
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

**5. Backend Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: plantdetection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: your-registry/plantdetection-backend:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: plantdetection-config
        - secretRef:
            name: plantdetection-secrets
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: plantdetection
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### Helm Deployment

**Chart.yaml:**
```yaml
apiVersion: v2
name: plantdetection
description: Plant Leaf Disease Detection System
type: application
version: 1.0.0

dependencies:
  - name: postgresql
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
  - name: redis
    version: 17.x.x
    repository: https://charts.bitnami.com/bitnami

values:
  replicaCount: 3
  
  image:
    repository: your-registry/plantdetection-backend
    tag: latest
    pullPolicy: Always
  
  service:
    type: ClusterIP
    port: 8000
  
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
      - host: api.plantdetection.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: plantdetection-tls
        hosts:
          - api.plantdetection.com
  
  resources:
    limits:
      cpu: 2
      memory: 8Gi
    requests:
      cpu: 1
      memory: 4Gi
  
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
```

**Deploy with Helm:**
```bash
# Add dependencies
helm repo add bitnami https://charts.bitnami.com/bitnami
helm dependency update

# Install
helm install plantdetection . \
  --namespace plantdetection \
  --create-namespace \
  --values production-values.yaml

# Upgrade
helm upgrade plantdetection . \
  --namespace plantdetection \
  --values production-values.yaml
```

## Cloud Platform Deployment

### AWS Deployment

**1. EKS Cluster Setup:**
```bash
# Create EKS cluster
eksctl create cluster \
  --name plantdetection \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name plantdetection
```

**2. Deploy with Terraform:**
```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# EKS Cluster
resource "aws_eks_cluster" "plantdetection" {
  name     = "plantdetection"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.24"
  
  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }
}

# Application Load Balancer
resource "aws_lb" "plantdetection" {
  name               = "plantdetection-alb"
  internal           = false
  load_balancer_type = "application"
  
  subnets = aws_subnet.public[*].id
  
  security_groups = [aws_security_group.plantdetection.id]
}

# ECS Service (alternative to EKS)
resource "aws_ecs_service" "backend" {
  name            = "plantdetection-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 3
  
  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }
}
```

### Google Cloud Deployment

**1. GKE Cluster:**
```bash
# Create GKE cluster
gcloud container clusters create plantdetection \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair

# Get credentials
gcloud container clusters get-credentials plantdetection \
  --zone us-central1-a
```

**2. Cloud Run Deployment:**
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT-ID/plantdetection-backend

# Deploy to Cloud Run
gcloud run deploy plantdetection-backend \
  --image gcr.io/PROJECT-ID/plantdetection-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1
```

### Azure Deployment

**1. AKS Cluster:**
```bash
# Create resource group
az group create --name plantdetection-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group plantdetection-rg \
  --name plantdetection \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group plantdetection-rg --name plantdetection
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'plantdetection-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Plant Detection System",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Analysis Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(analysis_success_total[5m]) / rate(analysis_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "Model Performance",
        "type": "table",
        "targets": [
          {
            "expr": "model_processing_time_seconds",
            "format": "table"
          }
        ]
      }
    ]
  }
}
```

## Security Considerations

### SSL/TLS Configuration

1. **Generate SSL certificates:**
```bash
# Let's Encrypt
certbot certonly --standalone -d api.plantdetection.com

# Self-signed (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem
```

2. **Nginx SSL Configuration:**
```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### API Security

1. **Rate Limiting:**
```yaml
# nginx rate limiting
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;
    
    server {
        location /api/v1/analyze {
            limit_req zone=api burst=20 nodelay;
            limit_req zone=upload burst=5 nodelay;
        }
    }
}
```

2. **Authentication Headers:**
```python
# backend/middleware/security.py
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-Key")
    if not api_key or not verify_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

def verify_key(api_key: str) -> bool:
    # Implement key verification logic
    return api_key in os.getenv("VALID_API_KEYS", "").split(",")
```

## Performance Optimization

### Database Optimization

```sql
-- PostgreSQL optimization
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_analysis_timestamp 
ON analysis_records (timestamp DESC);

CREATE INDEX CONCURRENTLY idx_analysis_plant_disease 
ON analysis_records (plant_name, disease_name);

-- Partition large tables by date
CREATE TABLE analysis_records_y2024m01 PARTITION OF analysis_records
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Optimize configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

### Caching Strategy

```python
# backend/cache/redis_cache.py
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expire=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(cache_key, expire, json.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expire=1800)
async def get_disease_info(disease_id: str):
    # Database query here
    pass
```

## Troubleshooting

### Common Issues

1. **High Memory Usage:**
```bash
# Check memory usage
docker stats --no-stream

# Restart services
docker-compose restart backend

# Scale resources
docker-compose up -d --scale backend=5
```

2. **Database Connection Issues:**
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d plantdetection -c "SELECT 1;"

# Reset connection pool
docker-compose restart backend
```

3. **API Performance Issues:**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/health"

# Monitor with Prometheus
curl "http://localhost:9090/api/v1/query?query=rate(http_request_duration_seconds_sum[5m])"
```

### Health Checks

```python
# scripts/health_check.py
import requests
import sys

def check_endpoint(url: str, name: str):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {name}: Healthy")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    endpoints = [
        ("http://localhost:8000/api/v1/health", "Backend API"),
        ("http://localhost:8501", "Frontend"),
        ("http://localhost:9090/-/healthy", "Prometheus"),
        ("http://localhost:3000/api/health", "Grafana")
    ]
    
    healthy = sum(check_endpoint(url, name) for url, name in endpoints)
    total = len(endpoints)
    
    print(f"\nHealth Status: {healthy}/{total} services healthy")
    return healthy == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
```

### Backup and Recovery

```bash
#!/bin/bash
# scripts/backup.sh

# Database backup
docker-compose exec postgres pg_dump -U postgres plantdetection > backup.sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE

# Model files backup
tar -czf models_backup.tar.gz models/

# Upload to cloud storage
aws s3 cp backup.sql s3://plantdetection-backups/
aws s3 cp models_backup.tar.gz s3://plantdetection-backups/

echo "Backup completed successfully"
```

This comprehensive deployment guide provides everything needed to deploy the Plant Leaf Disease Detection System in production, including Docker, Kubernetes, and major cloud platforms with proper monitoring, security, and performance optimization.