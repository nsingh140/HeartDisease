#!/usr/bin/env python3
"""
Deployment script for Heart Disease MLOps Project
"""

import subprocess
import sys
import os
import time
import requests

def run_command(command, check=True):
    """Run shell command and return result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def build_docker_image():
    """Build Docker image"""
    print("Building Docker image...")
    run_command("docker build -t heart-disease-api:latest .")
    print("Docker image built successfully!")

def test_docker_locally():
    """Test Docker container locally"""
    print("Testing Docker container locally...")
    
    # Stop any existing container
    run_command("docker stop heart-disease-test", check=False)
    run_command("docker rm heart-disease-test", check=False)
    
    # Run container
    run_command("docker run -d -p 8000:8000 --name heart-disease-test heart-disease-api:latest")
    
    # Wait for container to start
    time.sleep(10)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✓ Health check passed")
        else:
            print("✗ Health check failed")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test prediction endpoint
    test_data = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=test_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction test passed: {result}")
        else:
            print("✗ Prediction test failed")
            return False
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False
    
    # Cleanup
    run_command("docker stop heart-disease-test")
    run_command("docker rm heart-disease-test")
    
    return True

def deploy_to_kubernetes():
    """Deploy to Kubernetes"""
    print("Deploying to Kubernetes...")
    
    # Apply Kubernetes manifests
    run_command("kubectl apply -f k8s/deployment.yaml")
    run_command("kubectl apply -f k8s/service.yaml")
    
    print("Waiting for deployment to be ready...")
    run_command("kubectl wait --for=condition=available --timeout=300s deployment/heart-disease-api")
    
    # Get service info
    result = run_command("kubectl get services heart-disease-api-service")
    print("Service deployed:")
    print(result.stdout)
    
    print("✓ Kubernetes deployment completed!")

def deploy_monitoring():
    """Deploy monitoring stack"""
    print("Deploying monitoring stack...")
    run_command("kubectl apply -f k8s/monitoring.yaml")
    
    print("Waiting for monitoring to be ready...")
    time.sleep(30)
    
    result = run_command("kubectl get services prometheus-service grafana-service")
    print("Monitoring services:")
    print(result.stdout)
    
    print("✓ Monitoring deployment completed!")

def main():
    """Main deployment function"""
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [local|k8s|monitoring|all]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "local":
        build_docker_image()
        if test_docker_locally():
            print("✓ Local deployment successful!")
        else:
            print("✗ Local deployment failed!")
            sys.exit(1)
    
    elif mode == "k8s":
        build_docker_image()
        deploy_to_kubernetes()
    
    elif mode == "monitoring":
        deploy_monitoring()
    
    elif mode == "all":
        build_docker_image()
        if test_docker_locally():
            deploy_to_kubernetes()
            deploy_monitoring()
            print("✓ Full deployment completed!")
        else:
            print("✗ Local tests failed, skipping Kubernetes deployment")
            sys.exit(1)
    
    else:
        print("Invalid mode. Use: local, k8s, monitoring, or all")
        sys.exit(1)

if __name__ == "__main__":
    main()