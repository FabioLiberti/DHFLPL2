#!/bin/bash
##
# Setup cluster k3s per DHFLPL2
#
# Installa k3s sul nodo controller e configura il cluster
# per il Federated Learning su architetture eterogenee.
#
# Riferimento: Sezione 5.2 del paper
#
# Uso:
#   Controller:  ./setup_cluster.sh controller
#   Worker:      ./setup_cluster.sh worker <SERVER_IP> <TOKEN>
##

set -e

ROLE=${1:-controller}
SERVER_IP=${2:-""}
TOKEN=${3:-""}

echo "============================================"
echo "DHFLPL2 - Setup cluster k3s"
echo "Ruolo: $ROLE"
echo "============================================"

install_k3s_controller() {
    echo "Installazione k3s controller..."
    curl -sfL https://get.k3s.io | sh -

    echo "Attesa avvio k3s..."
    sleep 10

    echo "Verifica stato cluster:"
    sudo kubectl get nodes

    echo ""
    echo "Token per i worker:"
    sudo cat /var/lib/rancher/k3s/server/node-token

    echo ""
    echo "IP del controller (usare per i worker):"
    hostname -I | awk '{print $1}'
}

install_k3s_worker() {
    if [ -z "$SERVER_IP" ] || [ -z "$TOKEN" ]; then
        echo "Errore: specificare SERVER_IP e TOKEN"
        echo "Uso: $0 worker <SERVER_IP> <TOKEN>"
        exit 1
    fi

    echo "Installazione k3s worker..."
    echo "  Server: $SERVER_IP"

    curl -sfL https://get.k3s.io | \
        K3S_URL="https://${SERVER_IP}:6443" \
        K3S_TOKEN="$TOKEN" \
        sh -

    echo "Worker registrato al cluster."
}

install_rancher() {
    echo "Installazione Rancher..."

    sudo kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml

    echo "Attesa cert-manager..."
    sleep 30

    helm repo add rancher-latest https://releases.rancher.com/server-charts/latest
    helm repo update

    sudo kubectl create namespace cattle-system 2>/dev/null || true

    helm install rancher rancher-latest/rancher \
        --namespace cattle-system \
        --set hostname=rancher.local \
        --set replicas=1 \
        --set bootstrapPassword=admin

    echo ""
    echo "Rancher installato."
    echo "Accedi a: https://$(hostname -I | awk '{print $1}')"
}

deploy_fl() {
    echo "Deploy componenti Federated Learning..."

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    K8S_DIR="$SCRIPT_DIR/../deploy/k8s"

    sudo kubectl apply -f "$K8S_DIR/namespace.yml"
    sudo kubectl apply -f "$K8S_DIR/server-deployment.yml"
    sudo kubectl apply -f "$K8S_DIR/client-deployment.yml"
    sudo kubectl apply -f "$K8S_DIR/autoscaler.yml"

    echo "Verifica deployment:"
    sudo kubectl get all -n federated-learning
}

case $ROLE in
    controller)
        install_k3s_controller
        echo ""
        read -p "Installare Rancher? (y/n): " INSTALL_RANCHER
        if [ "$INSTALL_RANCHER" = "y" ]; then
            install_rancher
        fi
        echo ""
        read -p "Deployare FL components? (y/n): " DEPLOY_FL
        if [ "$DEPLOY_FL" = "y" ]; then
            deploy_fl
        fi
        ;;
    worker)
        install_k3s_worker
        ;;
    deploy)
        deploy_fl
        ;;
    *)
        echo "Uso: $0 {controller|worker|deploy}"
        echo ""
        echo "  controller              Installa k3s controller + opzionalmente Rancher"
        echo "  worker <IP> <TOKEN>     Registra un worker al cluster"
        echo "  deploy                  Deploya i componenti FL sul cluster"
        exit 1
        ;;
esac

echo ""
echo "Setup completato."
