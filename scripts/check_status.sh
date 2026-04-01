#!/bin/bash
##
# DHFLPL2 - Controllo stato piano sperimentale
#
# Mostra la tabella di stato di tutti gli esperimenti,
# indicando quali sono completati e quali mancano.
#
# Uso:
#   bash /Users/fxlybs/_DEV/DHFLPL2/scripts/check_status.sh
##

RESULTS_DIR="/Users/fxlybs/_DEV/DHFLPL2/experiments/results"
DEMOS_DIR="/Users/fxlybs/_DEV/DHFLPL2/demos/outputs"

DATASETS_STD=(cifar10 cifar100 mnist fashion_mnist svhn)
DATASETS_DP=(cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp)
CLIENTS=(2 5 10 20 50)

# Conta esperimenti
DONE_STD=0
DONE_DP=0
TOTAL_STD=25
TOTAL_DP=25

for ds in "${DATASETS_STD[@]}"; do
    for cl in "${CLIENTS[@]}"; do
        [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ] && DONE_STD=$((DONE_STD + 1))
    done
done
for ds in "${DATASETS_DP[@]}"; do
    for cl in "${CLIENTS[@]}"; do
        [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ] && DONE_DP=$((DONE_DP + 1))
    done
done

DONE_TOTAL=$((DONE_STD + DONE_DP))
PCT=$(( DONE_TOTAL * 100 / 50 ))

# Controlla grafici
PLOTS=0
[ -f "$RESULTS_DIR/figure2_accuracy_loss.png" ] && PLOTS=$((PLOTS + 1))
[ -f "$RESULTS_DIR/figure3_federated_vs_centralized.png" ] && PLOTS=$((PLOTS + 1))
[ -f "$RESULTS_DIR/summary_table.png" ] && PLOTS=$((PLOTS + 1))

# Controlla demo
DEMOS_DONE=0
DEMOS_LIST=(
    "dp_comparison_mnist.png"
    "gradient_inversion_attack.png"
    "membership_inference_attack.png"
    "model_update_leakage.png"
    "side_channel_analysis.png"
)
for d in "${DEMOS_LIST[@]}"; do
    [ -f "$DEMOS_DIR/$d" ] && DEMOS_DONE=$((DEMOS_DONE + 1))
done

# Processo in esecuzione?
RUNNING_PID=$(pgrep -f "experiments.run_experiment" 2>/dev/null || true)
RUNNING_INFO=""
if [ -n "$RUNNING_PID" ]; then
    RUNNING_INFO=$(ps -p $RUNNING_PID -o args= 2>/dev/null | sed 's/.*--config.*configs\///' | sed 's/\.yml.*//')
    RUNNING_STATUS="IN ESECUZIONE (PID $RUNNING_PID: $RUNNING_INFO)"
else
    RUNNING_STATUS="NESSUN PROCESSO ATTIVO"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  DHFLPL2 - STATO PIANO SPERIMENTALE                                    ║"
echo "║  $(date '+%Y-%m-%d %H:%M:%S')                                                       ║"
echo "║  Progresso: $DONE_TOTAL/50 esperimenti ($PCT%)                                      ║"
printf "║  Processo:  %-59s ║\n" "$RUNNING_STATUS"
echo "╠══════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                        ║"
echo "║  Fase 1 — Standard ($DONE_STD/$TOTAL_STD completati)                                    ║"
echo "║  ┌──────────────────┬────────┬────────┬────────┬────────┬────────┐     ║"
echo "║  │ Dataset          │  2 cl  │  5 cl  │ 10 cl  │ 20 cl  │ 50 cl  │     ║"
echo "║  ├──────────────────┼────────┼────────┼────────┼────────┼────────┤     ║"

for ds in "${DATASETS_STD[@]}"; do
    vals=()
    for cl in "${CLIENTS[@]}"; do
        if [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ]; then
            vals+=("done")
        else
            vals+=("-")
        fi
    done
    printf "║  │ %-16s │ %-6s │ %-6s │ %-6s │ %-6s │ %-6s │     ║\n" \
        "$ds" "${vals[0]}" "${vals[1]}" "${vals[2]}" "${vals[3]}" "${vals[4]}"
done

echo "║  └──────────────────┴────────┴────────┴────────┴────────┴────────┘     ║"
echo "║                                                                        ║"
echo "║  Fase 2 — Con DP ($DONE_DP/$TOTAL_DP completati)                                      ║"
echo "║  ┌──────────────────┬────────┬────────┬────────┬────────┬────────┐     ║"
echo "║  │ Dataset          │  2 cl  │  5 cl  │ 10 cl  │ 20 cl  │ 50 cl  │     ║"
echo "║  ├──────────────────┼────────┼────────┼────────┼────────┼────────┤     ║"

for ds in "${DATASETS_DP[@]}"; do
    vals=()
    for cl in "${CLIENTS[@]}"; do
        if [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ]; then
            vals+=("done")
        else
            vals+=("-")
        fi
    done
    printf "║  │ %-16s │ %-6s │ %-6s │ %-6s │ %-6s │ %-6s │     ║\n" \
        "$ds" "${vals[0]}" "${vals[1]}" "${vals[2]}" "${vals[3]}" "${vals[4]}"
done

echo "║  └──────────────────┴────────┴────────┴────────┴────────┴────────┘     ║"
echo "║                                                                        ║"

# Stato grafici
if [ $PLOTS -eq 3 ]; then
    PLOT_STATUS="done ($PLOTS/3 grafici)"
elif [ $PLOTS -gt 0 ]; then
    PLOT_STATUS="parziale ($PLOTS/3 grafici)"
else
    PLOT_STATUS="-"
fi

# Stato demo
if [ $DEMOS_DONE -eq 5 ]; then
    DEMO_STATUS="done ($DEMOS_DONE/5 grafici + 1 testuale)"
elif [ $DEMOS_DONE -gt 0 ]; then
    DEMO_STATUS="parziale ($DEMOS_DONE/5 grafici)"
else
    DEMO_STATUS="-"
fi

printf "║  Fase 3 — Grafici: %-49s ║\n" "$PLOT_STATUS"
printf "║  Fase 4 — 6 Demo:  %-49s ║\n" "$DEMO_STATUS"
echo "║                                                                        ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
