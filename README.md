# DHFLPL2 — Federated Learning in Dynamic and Heterogeneous Environments

> Advantages, Performances, and Privacy Problems

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fapp14188490-blue)](https://doi.org/10.3390/app14188490)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Journal](https://img.shields.io/badge/Applied%20Sciences-2024-green)](https://www.mdpi.com/journal/applsci)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue)](https://python.org)

## Overview

A distributed Federated Learning (FL) framework designed for dynamic and heterogeneous environments with IoT devices and Edge Computing infrastructures. The system implements FedAvg on heterogeneous architectures (ARM64 + x86_64) orchestrated via Kubernetes (k3s), with Differential Privacy support.

**Paper:** Liberti, F.; Berardi, D.; Martini, B. *Federated Learning in Dynamic and Heterogeneous Environments: Advantages, Performances, and Privacy Problems.* Applied Sciences **2024**, 14(18), 8490.

## Authors

**Fabio Liberti**, **Davide Berardi**\*, **Barbara Martini**

Department of Science and Engineering, Universitas Mercatorum, 00186 Rome, Italy

\* Correspondence: davide.berardi@unimercatorum.it

## Features

- **Federated Averaging (FedAvg)** — weighted aggregation across heterogeneous clients
- **5 benchmark datasets** — CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN
- **Non-IID data partitioning** — realistic federated data distribution
- **Differential Privacy** — Gaussian/Laplace noise mechanisms, gradient clipping, data redaction
- **Threat model simulation** — Gradient Inversion, Model Update Leakage, Membership Inference, Side-Channel analysis
- **Container deployment** — Docker + Kubernetes (k3s) with support for ARM64 and x86_64
- **Custom autoscaling** — precision-based automatic worker provisioning
- **Reproducible experiments** — YAML configs, CLI runner, automated plot generation

## Project Structure

```
DHFLPL2/
├── src/
│   ├── models/cnn.py              # CNN model for image classification
│   ├── data/
│   │   ├── loader.py              # Dataset loading (5 datasets)
│   │   └── partitioner.py         # Non-IID data distribution
│   ├── federation/
│   │   ├── server.py              # FedAvg server orchestration
│   │   ├── client.py              # FL client with local training
│   │   ├── strategy.py            # Aggregation strategies
│   │   ├── server_app.py          # Flower SuperLink entry point
│   │   └── client_app.py          # Flower SuperNode entry point
│   ├── privacy/
│   │   ├── dp_mechanism.py        # Differential privacy mechanisms
│   │   └── threat_model.py        # Attack vector simulation
│   ├── metrics/evaluation.py      # Precision, recall, F1, accuracy
│   └── utils/config.py            # Centralized configuration
├── deploy/
│   ├── docker/                    # Dockerfiles + docker-compose
│   └── k8s/                       # k3s manifests + autoscaler
├── experiments/
│   ├── configs/                   # YAML configs per dataset
│   ├── run_experiment.py          # Experiment runner CLI
│   └── results/                   # Output directory
├── scripts/
│   ├── setup_cluster.sh           # k3s cluster setup
│   ├── deploy.sh                  # Build + deploy automation
│   └── plot_results.py            # Figure generation
├── tests/                         # Unit tests
├── paper/                         # Published paper (PDF)
└── CHANGELOG.md                   # Development log
```

## Quick Start

### Installation

```bash
git clone https://github.com/FabioLiberti/DHFLPL2.git
cd DHFLPL2

# Conda (recommended)
conda env create -f environment.yml
conda activate dhflpl2
pip install -e .

# Alternatively, with pip
pip install -r requirements.txt
pip install -e .
```

### Run a Federated Experiment

```bash
# Single dataset, default clients (2, 5, 10, 20, 50)
python -m experiments.run_experiment --config experiments/configs/cifar10.yml

# Specific number of clients
python -m experiments.run_experiment --config experiments/configs/mnist.yml --clients 10

# All datasets, all configurations
python -m experiments.run_experiment --all
```

### Generate Plots

```bash
python scripts/plot_results.py --results-dir experiments/results/
```

### Docker (Local Testing)

```bash
cd deploy/docker
docker-compose up --build
```

### Kubernetes (k3s) Deployment

```bash
# Setup controller node
./scripts/setup_cluster.sh controller

# Add worker nodes
./scripts/setup_cluster.sh worker <CONTROLLER_IP> <TOKEN>

# Deploy FL components
./scripts/deploy.sh --clients 10 --dataset cifar10

# Scale workers
./scripts/deploy.sh --scale 50
```

## Privacy & Threat Model Demos

Executable demonstrations of the privacy mechanisms and attack vectors described in the paper (Section 4.2):

```bash
# Differential Privacy comparison (standard vs DP)
python -m demos.demo_dp_comparison --dataset mnist --rounds 30

# Gradient Inversion Attack simulation
python -m demos.demo_gradient_inversion --epsilon 1.0

# Membership Inference Attack analysis
python -m demos.demo_membership_inference --rounds 15

# Model Update Leakage monitoring
python -m demos.demo_model_update_leakage --rounds 20

# Side-Channel Attack analysis
python -m demos.demo_side_channel --clients 5

# Data Redaction pipeline (text, records, numerical noise)
python -m demos.demo_data_redaction
```

Results and plots are saved in `demos/outputs/`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Data (Clients)                  │
├─────────────┬─────────────┬─────────────┬───────────────┤
│  Flower     │  Flower     │  Flower     │  Flower       │
│  SuperNode  │  SuperNode  │  SuperNode  │  SuperNode    │
├─────────────┼─────────────┼─────────────┼───────────────┤
│  kubelet    │  kubelet    │  kubelet    │  kubelet      │
│ FL aggreg.  │ FL worker 1 │ FL worker 2 │ FL worker 3   │
├─────────────┴─────────────┴─────────────┴───────────────┤
│  k3s controller: Administrator │ API │ Dataset │ Rancher│
└─────────────────────────────────────────────────────────┘
```

Heterogeneous architecture support: ARM64 and x86_64 nodes in the same cluster, with automatic pod scheduling across architectures.

## Research Questions

| ID | Question |
|----|----------|
| **Q1** | What are the impacts of heterogeneous environments in Federated Learning? |
| **Q2** | How can the federated learning scenario benefit from different systems and architectures? |
| **Q3** | What are the privacy implications of using federated learning? |
| **Q4** | How can Kubernetes make heterogeneous ML models feasible and easier to conduct? |

## Results

Experiments across 5 datasets, 2 to 50 clients, 150 epochs:

| Metric | Best | Worst |
|--------|------|-------|
| **Accuracy** | 99.996% (MNIST, 2 clients) | 35.53% (CIFAR-100, 50 clients) |

Key findings:
- Client count does not significantly impact accuracy for simple datasets
- Class granularity is the dominant factor in accuracy degradation
- Heterogeneous architectures (ARM + x86) show comparable progression to homogeneous setups
- Custom autoscaling (precision threshold at 50%) improves model convergence

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python >= 3.10 |
| ML Framework | TensorFlow/Keras |
| FL Framework | [Flower](https://flower.ai/) |
| Privacy | Differential Privacy (Gaussian, Laplace) |
| Orchestration | Kubernetes (k3s) + Docker |
| Cloud | Oracle Cloud Infrastructure (OCI) |
| Management | Rancher |

## Citation

```bibtex
@article{Liberti2024,
  title     = {Federated Learning in Dynamic and Heterogeneous Environments:
               Advantages, Performances, and Privacy Problems},
  author    = {Liberti, Fabio and Berardi, Davide and Martini, Barbara},
  journal   = {Applied Sciences},
  volume    = {14},
  number    = {18},
  pages     = {8490},
  year      = {2024},
  publisher = {MDPI},
  doi       = {10.3390/app14188490}
}
```

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
