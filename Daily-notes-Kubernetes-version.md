
# Distributed GPU Cluster on Windows Laptops

This repository provides a comprehensive guide on how to utilize existing hardware, particularly Windows laptops, to build a distributed GPU cluster using Windows Subsystem for Linux (WSL). The repository covers two versions of cluster deployment: one using Docker Swarm and the other using Kubernetes deployed on WSL.

## Overview

The goal of this project is to demonstrate how to repurpose high-performance laptops, equipped with powerful GPUs, into a distributed computing cluster. This setup is ideal for machine learning tasks and other GPU-intensive computations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Version 1: Docker Swarm](#version-1-docker-swarm)
  - [Step 1: Install WSL and Ubuntu](#step-1-install-wsl-and-ubuntu)
  - [Step 2: Install Docker Desktop and NVIDIA Toolkit](#step-2-install-docker-desktop-and-nvidia-toolkit)
  - [Step 3: Set Up OpenMPI and Docker Image](#step-3-set-up-openmpi-and-docker-image)
  - [Step 4: Set Up Networking for the Cluster](#step-4-set-up-networking-for-the-cluster)
  - [Step 5: Run Distributed Workloads](#step-5-run-distributed-workloads)
  - [Step 6: Monitor and Optimize](#step-6-monitor-and-optimize)
- [Version 2: Kubernetes](#version-2-kubernetes)
  - [Step 1: Install WSL and Ubuntu](#step-1-install-wsl-and-ubuntu-1)
  - [Step 2: Install Docker Desktop and NVIDIA Toolkit](#step-2-install-docker-desktop-and-nvidia-toolkit-1)
  - [Step 3: Set Up Kubernetes](#step-3-set-up-kubernetes)
  - [Step 4: Deploy Applications](#step-4-deploy-applications)
  - [Step 5: Monitor and Optimize](#step-5-monitor-and-optimize-1)
- [Conclusion](#conclusion)

## Prerequisites

- Windows 10 or later
- WSL 2 with Ubuntu installed
- NVIDIA GPU with the latest drivers
- Docker Desktop
- Internet connection

## Version 1: Docker Swarm

### Step 1: Install WSL and Ubuntu

1. Open PowerShell as Administrator and enable WSL:
   powershell
   wsl --install
   
2. Set Ubuntu as the default distribution (if not already installed):
   powershell
   wsl --set-default Ubuntu
   

### Step 2: Install Docker Desktop and NVIDIA Toolkit

1. **Install Docker Desktop**
   - Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/).
   - In Docker Desktop settings, enable WSL 2 integration for Ubuntu.

2. **Install NVIDIA Drivers and Toolkit**
   - Ensure the latest NVIDIA drivers are installed.
   - Follow the [NVIDIA Container Toolkit for WSL](https://developer.nvidia.com/cuda/wsl) setup guide:
     bash
     sudo apt update
     sudo apt install -y nvidia-container-toolkit
     sudo systemctl restart docker
     

### Step 3: Set Up OpenMPI and Docker Image

1. **Install OpenMPI**:
   bash
   sudo apt update
   sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
   

2. **Create a Docker Image with OpenMPI and ML Framework**
   - Create a `Dockerfile`:
     ```Dockerfile
     FROM nvidia/cuda:12.2.0-base
     RUN apt-get update && apt-get install -y \
         python3 \
         python3-pip \
         openmpi-bin \
         libopenmpi-dev \
         && rm -rf /var/lib/apt/lists/*
     RUN pip3 install tensorflow torch
     CMD ["/bin/bash"]
     ```
   - Build the image:
     ```bash
     docker build -t ml-cluster .
     ```

### Step 4: Set Up Networking for the Cluster

1. **Initialize Docker Swarm on Laptop 1**:
   bash
   docker swarm init
   
   - Note the join token provided.

2. **Join Laptop 2 to the Swarm**:
   bash
   docker swarm join --token <your-swarm-token> <manager-ip>:2377
   

### Step 5: Run Distributed Workloads

1. **Prepare a Distributed Training Script (e.g., TensorFlow)**:
   python
   import tensorflow as tf
   import horovod.tensorflow as hvd

   hvd.init()
   gpus = tf.config.experimental.list_physical_devices('GPU')
   tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

   # Example: Simple MNIST training
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
   model = tf.keras.Sequential([
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10, activation='softmax')
   ])
   model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=5)
   

2. **Run the Script Across Both Laptops**:
   bash
   mpirun -np 2 -H laptop1-ip:1,laptop2-ip:1 python3 train.py
   

### Step 6: Monitor and Optimize

- Use `nvidia-smi` to monitor GPU utilization:
  bash
  watch -n 1 nvidia-smi
  
- Optimize batch sizes and communication overhead as needed.

## Version 2: Kubernetes

### Step 1: Install WSL and Ubuntu

1. Open PowerShell as Administrator and enable WSL:
   powershell
   wsl --install
   
2. Set Ubuntu as the default distribution (if not already installed):
   powershell
   wsl --set-default Ubuntu
   

### Step 2: Install Docker Desktop and NVIDIA Toolkit

1. **Install Docker Desktop**
   - Download and install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/).
   - In Docker Desktop settings, enable WSL 2 integration for Ubuntu.

2. **Install NVIDIA Drivers and Toolkit**
   - Ensure the latest NVIDIA drivers are installed.
   - Follow the [NVIDIA Container Toolkit for WSL](https://developer.nvidia.com/cuda/wsl) setup guide:
     ```bash
     sudo apt update
     sudo apt install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```

### Step 3: Set Up Kubernetes

1. **Install Minikube**:
   ```bash
   curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
   sudo install minikube-linux-amd64 /usr/local/bin/minikube
   ```
2. **Start Minikube**:
   ```bash
   minikube start --driver=docker
   ```

### Step 4: Deploy Applications

1. **Create a Kubernetes Deployment**:
   - Create a `deployment.yaml`:
     ```yaml
     apiVersion: apps/v1
     kind: Deployment
     metadata:
       name: ml-deployment
     spec:
       replicas: 2
       selector:
         matchLabels:
           app: ml-app
       template:
         metadata:
           labels:
             app: ml-app
         spec:
           containers:
           - name: ml-container
             image: ml-cluster
             resources:
               limits:
                 nvidia.com/gpu: 1
     ```
   - Apply the deployment:
     ```bash
     kubectl apply -f deployment.yaml
     ```

### Step 5: Monitor and Optimize

- Use `kubectl` to monitor the cluster:
  ```bash
  kubectl get pods
  ```
- Optimize resource requests and limits as needed.

## Conclusion

This repository demonstrates how to repurpose existing hardware to build a powerful distributed GPU cluster using WSL. By following the steps outlined for Docker Swarm and Kubernetes deployments, you can harness the combined power of your laptops' GPUs for intensive computational tasks.

For more information and detailed steps, please refer to the [Daily-notes.md](https://github.com/cmartinio/Distributed-GPU-Cluster-on-Windows-Laptops/blob/master/Daily-notes.md) file.
