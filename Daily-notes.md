
## Implementing a Distributed GPU Cluster on Windows Laptops

Before relocating to Switzerland, I faced the challenge of determining how to manage the extensive IT equipment I had accumulated over years in the tech industry. My goal was to consolidate my homelab into portable yet powerful and durable devices capable of handling workloads similar to those in my original setup.

After thorough consideration, I opted for two fully maxed-out ThinkPad P13 Gen 1 laptops. These systems replaced a homelab environment that previously consisted of multiple servers running a wide range of self-hosted applications, including virtualized firewalls, WordPress servers, Kubernetes clusters, and more.

Transitioning from a highly virtualized TrueNAS system operating within a hyper-converged infrastructure on vSphere to a pair of laptops proved to be a significant challenge. The shift became even more complex as I decided to delve fully into the realm of AI and large language model (LLM) training. However, with two laptops equipped with powerful GPUs that were underutilized, I developed a robust infrastructure tailored to my LLM training requirements.

Despite the initial difficulties, the results have been extraordinary. The streamlined setup not only meets but exceeds my expectations, demonstrating the capabilities of modern portable hardware in supporting high-performance workloads.
The solution was to combine the power of my two laptops with powerful GPUs ( NVIDIA RTX 5000s ) to use them as a cluster, while leveraging Docker, below is the steps to accomplish that: 


#### Step 1: Install WSL and Ubuntu
1. Open PowerShell as Administrator and enable WSL:
   powershell
   wsl --install
   
2. Set Ubuntu as the default distribution (if not already installed):
   powershell
   wsl --set-default Ubuntu
   

#### Step 2: Install Docker Desktop and NVIDIA Toolkit
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

#### Step 3: Set Up OpenMPI and Docker Image
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

#### Step 4: Set Up Networking for the Cluster
1. **Initialize Docker Swarm on Laptop 1**:
   ```bash
   docker swarm init
   ```
   - Note the join token provided.

2. **Join Laptop 2 to the Swarm**:
   ```bash
   docker swarm join --token <your-swarm-token> <manager-ip>:2377
   ```

#### Step 5: Run Distributed Workloads
1. **Prepare a Distributed Training Script (e.g., TensorFlow)**:
   ```python
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
   ```

2. **Run the Script Across Both Laptops**:
   ```bash
   mpirun -np 2 -H laptop1-ip:1,laptop2-ip:1 python3 train.py
   ```

#### Step 6: Monitor and Optimize
- Use `nvidia-smi` to monitor GPU utilization:
  ```bash
  watch -n 1 nvidia-smi
  ```
- Optimize batch sizes and communication overhead as needed.

---
This setup allowed me to harness the combined GPU power of your two laptops for machine learning tasks.



