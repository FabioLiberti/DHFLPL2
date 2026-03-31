#!/bin/bash
##
# DHFLPL2 - Esecuzione completa notturna
#
# Esegue tutti gli esperimenti, demo e genera tutti i grafici.
# Pensato per esecuzione notturna non presidiata.
#
# Uso:
#   cd /Users/fxlybs/_DEV/DHFLPL2
#   conda activate dhflpl2
#   bash scripts/run_all_overnight.sh 2>&1 | tee overnight_log.txt
##

set -e

PROJECT_DIR="/Users/fxlybs/_DEV/DHFLPL2"
cd "$PROJECT_DIR"

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "============================================================"
echo "DHFLPL2 - Esecuzione completa"
echo "Inizio: $START_TIME"
echo "============================================================"

# ---------------------------------------------------------------
# 1. ESPERIMENTI STANDARD (5 dataset x 5 client = 25 esperimenti)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "FASE 1: Esperimenti standard (senza DP)"
echo "============================================================"

RESULTS_DIR="$PROJECT_DIR/experiments/results"

for config in cifar10 cifar100 mnist fashion_mnist svhn; do
    for clients in 2 5 10 20 50; do
        RESULT_FILE="$RESULTS_DIR/${config}_${clients}clients.json"
        if [ -f "$RESULT_FILE" ]; then
            echo ""
            echo ">>> SKIP $config - $clients client (gia' eseguito: $RESULT_FILE)"
            continue
        fi
        echo ""
        echo ">>> $config - $clients client"
        python -m experiments.run_experiment \
            --config "$PROJECT_DIR/experiments/configs/${config}.yml" \
            --clients $clients
    done
done

# ---------------------------------------------------------------
# 2. ESPERIMENTI CON DIFFERENTIAL PRIVACY (5 dataset x 5 client)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "FASE 2: Esperimenti con Differential Privacy"
echo "============================================================"

for config in cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp; do
    for clients in 2 5 10 20 50; do
        RESULT_FILE="$RESULTS_DIR/${config}_${clients}clients.json"
        if [ -f "$RESULT_FILE" ]; then
            echo ""
            echo ">>> SKIP $config - $clients client (gia' eseguito: $RESULT_FILE)"
            continue
        fi
        echo ""
        echo ">>> $config - $clients client"
        python -m experiments.run_experiment \
            --config "$PROJECT_DIR/experiments/configs/${config}.yml" \
            --clients $clients
    done
done

# ---------------------------------------------------------------
# 3. GENERAZIONE GRAFICI (Figure 2, Figure 3, tabella)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "FASE 3: Generazione grafici"
echo "============================================================"

python "$PROJECT_DIR/scripts/plot_results.py" \
    --results-dir "$PROJECT_DIR/experiments/results/"

# ---------------------------------------------------------------
# 4. DEMO PRIVACY E THREAT MODEL
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "FASE 4: Demo privacy e threat model"
echo "============================================================"

echo ""
echo ">>> Demo: Data Redaction"
python -m demos.demo_data_redaction

echo ""
echo ">>> Demo: DP Comparison (MNIST, 30 round)"
python -m demos.demo_dp_comparison --dataset mnist --rounds 30 --clients 2

echo ""
echo ">>> Demo: Gradient Inversion Attack"
python -m demos.demo_gradient_inversion --epsilon 1.0 --iterations 300

echo ""
echo ">>> Demo: Membership Inference Attack"
python -m demos.demo_membership_inference --rounds 15 --clients 2

echo ""
echo ">>> Demo: Model Update Leakage"
python -m demos.demo_model_update_leakage --rounds 20 --clients 2

echo ""
echo ">>> Demo: Side-Channel Analysis"
python -m demos.demo_side_channel --clients 5 --rounds 10

# ---------------------------------------------------------------
# COMPLETATO
# ---------------------------------------------------------------
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "============================================================"
echo "ESECUZIONE COMPLETATA"
echo "Inizio: $START_TIME"
echo "Fine:   $END_TIME"
echo ""
echo "Risultati esperimenti: $PROJECT_DIR/experiments/results/"
echo "Grafici esperimenti:   $PROJECT_DIR/experiments/results/*.png"
echo "Grafici demo:          $PROJECT_DIR/demos/outputs/*.png"
echo "Log:                   $PROJECT_DIR/overnight_log.txt"
echo "============================================================"
