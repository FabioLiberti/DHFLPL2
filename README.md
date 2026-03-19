# Federated Learning in Dynamic and Heterogeneous Environments

> Advantages, Performances, and Privacy Problems

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fapp14188490-blue)](https://doi.org/10.3390/app14188490)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Journal](https://img.shields.io/badge/Applied%20Sciences-2024-green)](https://www.mdpi.com/journal/applsci)

## Authors

**Fabio Liberti**, **Davide Berardi**\*, **Barbara Martini**

Department of Science and Engineering, Universitas Mercatorum, 00186 Rome, Italy

\* Correspondence: davide.berardi@unimercatorum.it

## Abstract

Federated Learning (FL) represents a promising distributed learning methodology particularly suitable for dynamic and heterogeneous environments characterized by Internet of Things (IoT) devices and Edge Computing infrastructures. This work explores advanced techniques for dynamic model adaptation and heterogeneous data management in edge computing scenarios, proposing innovative solutions to improve the robustness and efficiency of federated learning. We present an innovative solution based on Kubernetes (k3s) which enables the fast application of FL models to heterogeneous architectures (ARM and x86). Experimental results demonstrate that our proposals can improve the performance of FL in IoT and edge environments, offering new perspectives for the practical implementation of decentralized intelligent systems.

## Research Questions

| ID | Question |
|----|----------|
| **Q1** | What are the impacts of heterogeneous environments in Federated Learning? |
| **Q2** | How can the federated learning scenario benefit from different systems and architectures? |
| **Q3** | What are the privacy implications of using federated learning? |
| **Q4** | How can a cluster management system (i.e., Kubernetes) make the application of heterogeneous machine learning models feasible and easier to conduct? |

## Methodology

### Federated Averaging (FedAvg)

The system implements the FedAvg algorithm, where the global model is updated as:

```
w_{t+1} = Σ_{k=1}^{K} (n_k / n) * w_{t+1}^k
```

where *K* is the total number of participating devices, *n_k* is the number of data samples on device *k*, *n* is the total number of samples across all devices, and *w_{t+1}^k* is the weight vector of the local model updated by device *k* at round *t+1*.

### Differential Privacy

Privacy is enforced through (ε, δ)-differential privacy:

```
Pr[M(D) ∈ S] ≤ e^ε * Pr[M(D') ∈ S] + δ
```

Implemented via the `pydp` library with two actions:
1. **Redaction** of private data (email addresses, phone numbers, home addresses)
2. **Noise augmentation** to prevent recognition of private numerical values

### Privacy Threat Model

The system addresses the following attack vectors:
- **Gradient Inversion Attacks** — reconstruction of original data from shared gradients
- **Model Update Leakage** — inference from repeatedly shared model updates
- **Side-Channel Attacks** — exploitation of timing or communication size metadata
- **Membership Inference Attacks** — determining whether a data point was part of the training set

## Architecture

The system is built on a multilayer architecture using Docker and Kubernetes (k3s):

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

### Components

| Component | Role |
|-----------|------|
| **k3s Controller** | Lightweight Kubernetes distribution managing cluster orchestration |
| **Rancher** | Web-based Kubernetes management platform |
| **Flower (SuperLink + SuperNode)** | Federated learning framework handling model aggregation and distribution |
| **Docker Containers** | Isolated, reproducible environments per node |
| **Oracle Cloud Infrastructure (OCI)** | Hosting provider with x86 and ARM64 virtual machines |

### Heterogeneous Architecture Support

The platform runs on mixed architectures within the same cluster:
- **ARM64** — ARM-based virtual machines (Oracle Cloud)
- **x86_64 (AMD64)** — Intel-based virtual machines

Nodes auto-register to the Flower platform upon container startup, enabling dynamic scaling via:
```bash
kubectl scale --replicas=50 fl/client
```

Custom autoscaling is also implemented: when precision drops below 50%, new worker nodes are automatically provisioned.

## Datasets

Experiments were conducted across five benchmark datasets with 2 to 50 federated clients over 150 epochs:

| Dataset | Classes | Description | Source |
|---------|---------|-------------|--------|
| **CIFAR-10** | 10 | Color images (primary benchmark) | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **CIFAR-100** | 100 | Fine-grained color images | [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **MNIST** | 10 | Handwritten digits | [yann.lecun.com](http://yann.lecun.com/exdb/mnist/) |
| **Fashion-MNIST** | 10 | Clothing items | [fashion-mnist](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) |
| **SVHN** | 10 | House street numbers (Google Street View) | [stanford.edu](http://ufldl.stanford.edu/housenumbers/) |

Data distribution is **non-IID** across nodes, shuffled randomly to avoid overfitting.

## Results

| Metric | Best | Worst |
|--------|------|-------|
| **Accuracy** | 99.996% (MNIST, 2 clients) | 35.53% (CIFAR-100, 50 clients) |

### Key Findings

- **Client count** does not significantly impact accuracy for simple datasets (e.g., MNIST)
- **Class granularity** is the dominant factor in accuracy degradation (CIFAR-100 vs CIFAR-10)
- Performance on heterogeneous architectures (ARM + x86) shows **comparable progression** to homogeneous setups, without significant performance loss
- Federated accuracy remains **below centralized baselines** (~99%), consistent with known FL limitations due to local minima in distributed optimization

## Dimensions of Heterogeneity Addressed

| Dimension | Challenge |
|-----------|-----------|
| **Communication** | Variable bandwidth, latency, reliability across network protocols |
| **Models** | Different architectures and data formats affecting aggregation |
| **Statistics** | Non-uniform, non-IID data distributions causing over/under-fitting |
| **Devices** | Disparate computing power, memory, and communication capabilities |

## Technology Stack

- **Orchestration**: Kubernetes (k3s) + Docker
- **Federated Learning**: [Flower](https://flower.ai/) (SuperLink + SuperNode architecture)
- **Privacy**: `pydp` (differential privacy)
- **Cloud**: Oracle Cloud Infrastructure (OCI)
- **Management**: Rancher
- **ML Framework**: TensorFlow/Keras
- **Language**: Python

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
