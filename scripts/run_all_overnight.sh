#!/bin/bash
##
# DHFLPL2 - Esecuzione completa notturna
#
# Esegue tutti gli esperimenti, demo e genera tutti i grafici.
# Pensato per esecuzione notturna non presidiata.
# Mostra tabella di sintesi con progresso e tempi.
#
# Uso:
#   cd /Users/fxlybs/_DEV/DHFLPL2
#   conda activate dhflpl2
#   bash scripts/run_all_overnight.sh 2>&1 | tee overnight_log.txt
##

set -e

PROJECT_DIR="/Users/fxlybs/_DEV/DHFLPL2"
cd "$PROJECT_DIR"
RESULTS_DIR="$PROJECT_DIR/experiments/results"

# Contatori globali
TOTAL_EXPERIMENTS=50
TOTAL_TASKS=58  # 50 esperimenti + 1 grafici + 6 demo + 1 summary
COMPLETED=0
SKIPPED=0
TASK_TIMES=()

START_TIME=$(date '+%s')
START_TIME_FMT=$(date '+%Y-%m-%d %H:%M:%S')

print_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  DHFLPL2 - Piano Sperimentale Completo                     ║"
    echo "║  Inizio: $START_TIME_FMT                              ║"
    echo "║  Totale previsto: $TOTAL_TASKS task (50 exp + 1 plot + 6 demo + 1 summary) ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

print_progress() {
    local current=$1
    local total=$2
    local description="$3"
    local elapsed=$(( $(date '+%s') - START_TIME ))
    local elapsed_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

    local pct=0
    if [ $current -gt 0 ]; then
        pct=$(( current * 100 / total ))
    fi
    local bar_len=30
    local filled=$(( pct * bar_len / 100 ))
    local empty=$(( bar_len - filled ))
    local bar=$(printf '█%.0s' $(seq 1 $filled 2>/dev/null) ; printf '░%.0s' $(seq 1 $empty 2>/dev/null))

    # Stima tempo rimanente
    local eta="--:--:--"
    if [ $current -gt 0 ] && [ $current -lt $total ]; then
        local avg=$(( elapsed / current ))
        local remaining=$(( avg * (total - current) ))
        eta=$(printf '%02d:%02d:%02d' $((remaining/3600)) $(((remaining%3600)/60)) $((remaining%60)))
    fi

    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│ [$bar] $pct%  ($current/$total)"
    echo "│ Attuale:   $description"
    echo "│ Trascorso: $elapsed_fmt  |  Completati: $COMPLETED  |  Saltati: $SKIPPED"
    echo "│ ETA stima: $eta"
    echo "└──────────────────────────────────────────────────────────────┘"
}

run_experiment() {
    local config_name="$1"
    local config_file="$2"
    local clients="$3"
    local task_num="$4"
    local result_file="$RESULTS_DIR/${config_name}_${clients}clients.json"

    if [ -f "$result_file" ]; then
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        echo "  >> SKIP $config_name - $clients client (gia' completato)"
        return
    fi

    print_progress $task_num $TOTAL_TASKS "$config_name | $clients client"

    local exp_start=$(date '+%s')

    python -m experiments.run_experiment \
        --config "$PROJECT_DIR/experiments/configs/${config_file}.yml" \
        --clients $clients

    local exp_end=$(date '+%s')
    local exp_elapsed=$(( exp_end - exp_start ))
    local exp_fmt=$(printf '%02d:%02d' $((exp_elapsed/60)) $((exp_elapsed%60)))
    TASK_TIMES+=("$config_name|${clients}cl|${exp_fmt}")
    COMPLETED=$((COMPLETED + 1))

    echo "  >> Completato in $exp_fmt ($config_name $clients client)"
}

print_summary_table() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  TABELLA RIEPILOGATIVA TEMPI DI ESECUZIONE                 ║"
    echo "╠════════════════════════════╦════════════╦═══════════════════╣"
    echo "║ Esperimento                ║  Client    ║  Tempo            ║"
    echo "╠════════════════════════════╬════════════╬═══════════════════╣"

    for entry in "${TASK_TIMES[@]}"; do
        IFS='|' read -r name clients time <<< "$entry"
        printf "║ %-26s ║ %-10s ║ %-17s ║\n" "$name" "$clients" "$time"
    done

    echo "╚════════════════════════════╩════════════╩═══════════════════╝"
}

# ===================================================================
# INIZIO ESECUZIONE
# ===================================================================

print_header

# ---------------------------------------------------------------
# FASE 1: ESPERIMENTI STANDARD (25)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  FASE 1/4: Esperimenti standard (senza DP) [25 esperimenti]"
echo "============================================================"

TASK_NUM=0
for config in cifar10 cifar100 mnist fashion_mnist svhn; do
    for clients in 2 5 10 20 50; do
        TASK_NUM=$((TASK_NUM + 1))
        run_experiment "$config" "$config" "$clients" "$TASK_NUM"
    done
done

# ---------------------------------------------------------------
# FASE 2: ESPERIMENTI CON DP (25)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  FASE 2/4: Esperimenti con Differential Privacy [25 esperimenti]"
echo "============================================================"

for config in cifar10 cifar100 mnist fashion_mnist svhn; do
    for clients in 2 5 10 20 50; do
        TASK_NUM=$((TASK_NUM + 1))
        run_experiment "${config}_dp" "${config}_dp" "$clients" "$TASK_NUM"
    done
done

# ---------------------------------------------------------------
# FASE 3: GENERAZIONE GRAFICI
# ---------------------------------------------------------------
TASK_NUM=$((TASK_NUM + 1))
print_progress $TASK_NUM $TOTAL_TASKS "Generazione grafici (Figure 2, 3, tabella)"

echo ""
echo "============================================================"
echo "  FASE 3/4: Generazione grafici"
echo "============================================================"

python "$PROJECT_DIR/scripts/plot_results.py" \
    --results-dir "$PROJECT_DIR/experiments/results/"

COMPLETED=$((COMPLETED + 1))

# ---------------------------------------------------------------
# FASE 4: DEMO PRIVACY E THREAT MODEL
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  FASE 4/4: Demo privacy e threat model [6 demo]"
echo "============================================================"

DEMOS=(
    "demos.demo_data_redaction|Data Redaction|"
    "demos.demo_dp_comparison|DP Comparison|--dataset mnist --rounds 30 --clients 2"
    "demos.demo_gradient_inversion|Gradient Inversion Attack|--epsilon 1.0 --iterations 300"
    "demos.demo_membership_inference|Membership Inference Attack|--rounds 15 --clients 2"
    "demos.demo_model_update_leakage|Model Update Leakage|--rounds 20 --clients 2"
    "demos.demo_side_channel|Side-Channel Analysis|--clients 5 --rounds 10"
)

for demo_entry in "${DEMOS[@]}"; do
    IFS='|' read -r module name args <<< "$demo_entry"
    TASK_NUM=$((TASK_NUM + 1))
    print_progress $TASK_NUM $TOTAL_TASKS "Demo: $name"

    local_start=$(date '+%s')
    python -m $module $args
    local_end=$(date '+%s')
    local_elapsed=$(( local_end - local_start ))
    local_fmt=$(printf '%02d:%02d' $((local_elapsed/60)) $((local_elapsed%60)))

    TASK_TIMES+=("Demo: $name||${local_fmt}")
    COMPLETED=$((COMPLETED + 1))

    echo "  >> Demo completata in $local_fmt"
done

# ---------------------------------------------------------------
# COMPLETATO
# ---------------------------------------------------------------
END_TIME=$(date '+%s')
END_TIME_FMT=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
TOTAL_FMT=$(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60)))

print_summary_table

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ESECUZIONE COMPLETATA                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Inizio:      $START_TIME_FMT                         ║"
echo "║  Fine:        $END_TIME_FMT                         ║"
echo "║  Durata:      $TOTAL_FMT                                  ║"
echo "║  Completati:  $COMPLETED task                                    ║"
echo "║  Saltati:     $SKIPPED task                                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Risultati:   experiments/results/*.json                   ║"
echo "║  Grafici:     experiments/results/*.png                    ║"
echo "║  Demo:        demos/outputs/*.png                          ║"
echo "║  Log:         overnight_log.txt                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
