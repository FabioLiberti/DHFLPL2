#!/bin/bash
##
# DHFLPL2 - Esecuzione completa notturna
#
# Esegue tutti gli esperimenti, demo e genera tutti i grafici.
# Pensato per esecuzione notturna non presidiata.
# Mostra tabella di stato e sintesi con progresso e tempi.
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

# Stato esperimenti: "done", "running", "skip", "-"
declare -A EXP_STATUS
DATASETS_STD=(cifar10 cifar100 mnist fashion_mnist svhn)
DATASETS_DP=(cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp)
CLIENTS=(2 5 10 20 50)
CURRENT_PHASE=0
PHASE_NAMES=("Non iniziata" "Fase 1: Standard" "Fase 2: Con DP" "Fase 3: Grafici" "Fase 4: Demo" "Completata")

# Inizializza stato
for ds in "${DATASETS_STD[@]}"; do
    for cl in "${CLIENTS[@]}"; do
        if [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ]; then
            EXP_STATUS["std_${ds}_${cl}"]="done"
        else
            EXP_STATUS["std_${ds}_${cl}"]="-"
        fi
    done
done
for ds in "${DATASETS_DP[@]}"; do
    for cl in "${CLIENTS[@]}"; do
        if [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ]; then
            EXP_STATUS["dp_${ds}_${cl}"]="done"
        else
            EXP_STATUS["dp_${ds}_${cl}"]="-"
        fi
    done
done

START_TIME=$(date '+%s')
START_TIME_FMT=$(date '+%Y-%m-%d %H:%M:%S')

print_status_table() {
    local elapsed=$(( $(date '+%s') - START_TIME ))
    local elapsed_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))
    local done_count=0
    local total_count=50

    # Conta completati
    for key in "${!EXP_STATUS[@]}"; do
        if [ "${EXP_STATUS[$key]}" = "done" ] || [ "${EXP_STATUS[$key]}" = "skip" ]; then
            done_count=$((done_count + 1))
        fi
    done
    local pct=$(( done_count * 100 / total_count ))

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║  DHFLPL2 - STATO PIANO SPERIMENTALE                                    ║"
    echo "║  Trascorso: $elapsed_fmt  |  Progresso: $done_count/$total_count esperimenti ($pct%)            ║"
    echo "╠══════════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                        ║"
    echo "║  Fase 1 — Standard (25 esperimenti)                                    ║"
    echo "║  ┌──────────────────┬────────┬────────┬────────┬────────┬────────┐     ║"
    echo "║  │ Dataset          │  2 cl  │  5 cl  │ 10 cl  │ 20 cl  │ 50 cl  │     ║"
    echo "║  ├──────────────────┼────────┼────────┼────────┼────────┼────────┤     ║"

    for ds in "${DATASETS_STD[@]}"; do
        local c2="${EXP_STATUS[std_${ds}_2]}"
        local c5="${EXP_STATUS[std_${ds}_5]}"
        local c10="${EXP_STATUS[std_${ds}_10]}"
        local c20="${EXP_STATUS[std_${ds}_20]}"
        local c50="${EXP_STATUS[std_${ds}_50]}"
        printf "║  │ %-16s │ %-6s │ %-6s │ %-6s │ %-6s │ %-6s │     ║\n" \
            "$ds" "$c2" "$c5" "$c10" "$c20" "$c50"
    done

    echo "║  └──────────────────┴────────┴────────┴────────┴────────┴────────┘     ║"
    echo "║                                                                        ║"
    echo "║  Fase 2 — Con DP (25 esperimenti)                                      ║"
    echo "║  ┌──────────────────┬────────┬────────┬────────┬────────┬────────┐     ║"
    echo "║  │ Dataset          │  2 cl  │  5 cl  │ 10 cl  │ 20 cl  │ 50 cl  │     ║"
    echo "║  ├──────────────────┼────────┼────────┼────────┼────────┼────────┤     ║"

    for ds in "${DATASETS_DP[@]}"; do
        local c2="${EXP_STATUS[dp_${ds}_2]}"
        local c5="${EXP_STATUS[dp_${ds}_5]}"
        local c10="${EXP_STATUS[dp_${ds}_10]}"
        local c20="${EXP_STATUS[dp_${ds}_20]}"
        local c50="${EXP_STATUS[dp_${ds}_50]}"
        printf "║  │ %-16s │ %-6s │ %-6s │ %-6s │ %-6s │ %-6s │     ║\n" \
            "$ds" "$c2" "$c5" "$c10" "$c20" "$c50"
    done

    echo "║  └──────────────────┴────────┴────────┴────────┴────────┴────────┘     ║"
    echo "║                                                                        ║"

    # Stato fasi
    local f3_status="-"
    local f4_status="-"
    if [ $CURRENT_PHASE -ge 3 ]; then f3_status="in corso"; fi
    if [ $CURRENT_PHASE -ge 4 ]; then f3_status="done"; f4_status="in corso"; fi
    if [ $CURRENT_PHASE -ge 5 ]; then f4_status="done"; fi

    printf "║  Fase 3 — Grafici: %-49s ║\n" "$f3_status"
    printf "║  Fase 4 — 6 Demo:  %-49s ║\n" "$f4_status"
    echo "║                                                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
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
    local phase_prefix="$5"
    local result_file="$RESULTS_DIR/${config_name}_${clients}clients.json"

    if [ -f "$result_file" ]; then
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        EXP_STATUS["${phase_prefix}_${config_name}_${clients}"]="skip"
        echo "  >> SKIP $config_name - $clients client (gia' completato)"
        return
    fi

    EXP_STATUS["${phase_prefix}_${config_name}_${clients}"]=">>>"
    print_status_table
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

    EXP_STATUS["${phase_prefix}_${config_name}_${clients}"]="done"

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

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DHFLPL2 - Piano Sperimentale Completo                     ║"
echo "║  Inizio: $START_TIME_FMT                              ║"
echo "║  Totale: $TOTAL_TASKS task (50 exp + 1 plot + 6 demo + 1 summary)   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Stampa stato iniziale
print_status_table

# ---------------------------------------------------------------
# FASE 1: ESPERIMENTI STANDARD (25)
# ---------------------------------------------------------------
CURRENT_PHASE=1
echo ""
echo "============================================================"
echo "  FASE 1/4: Esperimenti standard (senza DP) [25 esperimenti]"
echo "============================================================"

TASK_NUM=0
for config in "${DATASETS_STD[@]}"; do
    for clients in "${CLIENTS[@]}"; do
        TASK_NUM=$((TASK_NUM + 1))
        run_experiment "$config" "$config" "$clients" "$TASK_NUM" "std"
    done
done

# ---------------------------------------------------------------
# FASE 2: ESPERIMENTI CON DP (25)
# ---------------------------------------------------------------
CURRENT_PHASE=2
echo ""
echo "============================================================"
echo "  FASE 2/4: Esperimenti con Differential Privacy [25 esperimenti]"
echo "============================================================"

for config in "${DATASETS_STD[@]}"; do
    dp_config="${config}_dp"
    for clients in "${CLIENTS[@]}"; do
        TASK_NUM=$((TASK_NUM + 1))
        run_experiment "$dp_config" "$dp_config" "$clients" "$TASK_NUM" "dp"
    done
done

# ---------------------------------------------------------------
# FASE 3: GENERAZIONE GRAFICI
# ---------------------------------------------------------------
CURRENT_PHASE=3
TASK_NUM=$((TASK_NUM + 1))
print_status_table
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
CURRENT_PHASE=4
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
CURRENT_PHASE=5
END_TIME=$(date '+%s')
END_TIME_FMT=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
TOTAL_FMT=$(printf '%02d:%02d:%02d' $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60)))

# Tabella stato finale
print_status_table

# Tabella tempi
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
