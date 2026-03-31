#!/bin/bash
##
# Deploy rapido su k3s per DHFLPL2
#
# Builda le immagini Docker e le deploya sul cluster k3s.
#
# Uso:
#   ./deploy.sh                    # Build + deploy default (2 client)
#   ./deploy.sh --clients 10       # Deploy con 10 client
#   ./deploy.sh --scale 50         # Scala a 50 client
#   ./deploy.sh --dataset mnist    # Cambia dataset
##

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
CLIENTS=2
DATASET="cifar10"
ACTION="deploy"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clients)
            CLIENTS="$2"
            shift 2
            ;;
        --scale)
            ACTION="scale"
            CLIENTS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --build-only)
            ACTION="build"
            shift
            ;;
        *)
            echo "Opzione sconosciuta: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "DHFLPL2 - Deploy"
echo "  Azione:   $ACTION"
echo "  Dataset:  $DATASET"
echo "  Client:   $CLIENTS"
echo "============================================"

build_images() {
    echo "Build immagini Docker..."
    cd "$PROJECT_DIR"

    docker build -t dhflpl2/server:latest -f deploy/docker/Dockerfile.server .
    docker build -t dhflpl2/client:latest -f deploy/docker/Dockerfile.client .

    echo "Immagini buildate."
}

deploy_k8s() {
    echo "Applicazione manifesti k3s..."

    sudo kubectl apply -f "$PROJECT_DIR/deploy/k8s/namespace.yml"

    # Aggiorna il dataset nei deployment
    sudo kubectl set env deployment/fl-server \
        FL_DATASET="$DATASET" \
        -n federated-learning

    sudo kubectl apply -f "$PROJECT_DIR/deploy/k8s/server-deployment.yml"

    sudo kubectl set env deployment/fl-client \
        FL_DATASET="$DATASET" \
        -n federated-learning

    sudo kubectl apply -f "$PROJECT_DIR/deploy/k8s/client-deployment.yml"

    # Scala al numero di client richiesto
    sudo kubectl scale deployment fl-client \
        --replicas="$CLIENTS" \
        -n federated-learning

    sudo kubectl apply -f "$PROJECT_DIR/deploy/k8s/autoscaler.yml"

    echo ""
    echo "Stato deployment:"
    sudo kubectl get pods -n federated-learning
}

scale_clients() {
    echo "Scaling client a $CLIENTS repliche..."
    sudo kubectl scale deployment fl-client \
        --replicas="$CLIENTS" \
        -n federated-learning

    echo "Stato:"
    sudo kubectl get pods -n federated-learning
}

case $ACTION in
    deploy)
        build_images
        deploy_k8s
        ;;
    build)
        build_images
        ;;
    scale)
        scale_clients
        ;;
esac

echo ""
echo "Completato."
