#!/bin/bash
##
# DHFLPL2 - Esecuzione completa notturna
#
# Esegue tutti gli esperimenti, demo e genera tutti i grafici.
# Pensato per esecuzione notturna non presidiata.
# Compatibile con bash 3+ (macOS).
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

TOTAL_TASKS=58
COMPLETED=0
SKIPPED=0
TASK_NUM=0
CURRENT_PHASE=0

START_TIME=$(date '+%s')
START_TIME_FMT=$(date '+%Y-%m-%d %H:%M:%S')

# File temporaneo per tracciare i tempi
TIMES_FILE=$(mktemp /tmp/dhflpl2_times.XXXXXX)
trap "rm -f $TIMES_FILE" EXIT

get_status() {
    local file="$RESULTS_DIR/${1}_${2}clients.json"
    if [ -f "$file" ]; then
        echo "done"
    else
        echo "-"
    fi
}

print_status_table() {
    local elapsed=$(( $(date '+%s') - START_TIME ))
    local elapsed_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

    local done_count=0
    for ds in cifar10 cifar100 mnist fashion_mnist svhn cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp; do
        for cl in 2 5 10 20 50; do
            [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ] && done_count=$((done_count + 1))
        done
    done
    local pct=$(( done_count * 100 / 50 ))

    local done_std=0
    for ds in cifar10 cifar100 mnist fashion_mnist svhn; do
        for cl in 2 5 10 20 50; do
            [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ] && done_std=$((done_std + 1))
        done
    done

    local done_dp=0
    for ds in cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp; do
        for cl in 2 5 10 20 50; do
            [ -f "$RESULTS_DIR/${ds}_${cl}clients.json" ] && done_dp=$((done_dp + 1))
        done
    done

    echo ""
    echo "========================================================================"
    echo "  DHFLPL2 - STATO PIANO SPERIMENTALE"
    echo "  Trascorso: $elapsed_fmt  |  Progresso: $done_count/50 esperimenti ($pct%)"
    echo "========================================================================"
    echo ""
    echo "  Fase 1 - Standard ($done_std/25 completati)"
    echo "  +------------------+--------+--------+--------+--------+--------+"
    echo "  | Dataset          |  2 cl  |  5 cl  | 10 cl  | 20 cl  | 50 cl  |"
    echo "  +------------------+--------+--------+--------+--------+--------+"

    for ds in cifar10 cifar100 mnist fashion_mnist svhn; do
        local c2=$(get_status "$ds" 2)
        local c5=$(get_status "$ds" 5)
        local c10=$(get_status "$ds" 10)
        local c20=$(get_status "$ds" 20)
        local c50=$(get_status "$ds" 50)

        # Marca in corso
        if [ "$CURRENT_RUNNING" = "${ds}" ]; then
            eval "c${CURRENT_CLIENTS}='>>>'"
        fi

        printf "  | %-16s | %-6s | %-6s | %-6s | %-6s | %-6s |\n" \
            "$ds" "$c2" "$c5" "$c10" "$c20" "$c50"
    done

    echo "  +------------------+--------+--------+--------+--------+--------+"
    echo ""
    echo "  Fase 2 - Con DP ($done_dp/25 completati)"
    echo "  +------------------+--------+--------+--------+--------+--------+"
    echo "  | Dataset          |  2 cl  |  5 cl  | 10 cl  | 20 cl  | 50 cl  |"
    echo "  +------------------+--------+--------+--------+--------+--------+"

    for ds in cifar10_dp cifar100_dp mnist_dp fashion_mnist_dp svhn_dp; do
        local c2=$(get_status "$ds" 2)
        local c5=$(get_status "$ds" 5)
        local c10=$(get_status "$ds" 10)
        local c20=$(get_status "$ds" 20)
        local c50=$(get_status "$ds" 50)

        if [ "$CURRENT_RUNNING" = "${ds}" ]; then
            eval "c${CURRENT_CLIENTS}='>>>'"
        fi

        printf "  | %-16s | %-6s | %-6s | %-6s | %-6s | %-6s |\n" \
            "$ds" "$c2" "$c5" "$c10" "$c20" "$c50"
    done

    echo "  +------------------+--------+--------+--------+--------+--------+"
    echo ""

    local f3="-"; local f4="-"
    [ $CURRENT_PHASE -ge 3 ] && f3="in corso"
    [ $CURRENT_PHASE -ge 4 ] && f3="done" && f4="in corso"
    [ $CURRENT_PHASE -ge 5 ] && f4="done"

    echo "  Fase 3 - Grafici: $f3"
    echo "  Fase 4 - 6 Demo:  $f4"
    echo "========================================================================"
}

print_progress() {
    local current=$1
    local total=$2
    local description="$3"
    local elapsed=$(( $(date '+%s') - START_TIME ))
    local elapsed_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)))

    local pct=0
    [ $current -gt 0 ] && pct=$(( current * 100 / total ))

    local eta="--:--:--"
    if [ $current -gt 0 ] && [ $current -lt $total ]; then
        local avg=$(( elapsed / current ))
        local remaining=$(( avg * (total - current) ))
        eta=$(printf '%02d:%02d:%02d' $((remaining/3600)) $(((remaining%3600)/60)) $((remaining%60)))
    fi

    echo ""
    echo "  [$pct%] ($current/$total) $description"
    echo "  Trascorso: $elapsed_fmt  |  Completati: $COMPLETED  |  Saltati: $SKIPPED  |  ETA: $eta"
}

run_experiment() {
    local config_name="$1"
    local config_file="$2"
    local clients="$3"
    local result_file="$RESULTS_DIR/${config_name}_${clients}clients.json"

    TASK_NUM=$((TASK_NUM + 1))

    if [ -f "$result_file" ]; then
        SKIPPED=$((SKIPPED + 1))
        COMPLETED=$((COMPLETED + 1))
        echo "  >> SKIP $config_name - $clients client (gia' completato)"
        return
    fi

    CURRENT_RUNNING="$config_name"
    CURRENT_CLIENTS="$clients"
    print_status_table
    print_progress $TASK_NUM $TOTAL_TASKS "$config_name | $clients client"

    local exp_start=$(date '+%s')

    python -m experiments.run_experiment \
        --config "$PROJECT_DIR/experiments/configs/${config_file}.yml" \
        --clients $clients

    local exp_end=$(date '+%s')
    local exp_elapsed=$(( exp_end - exp_start ))
    local exp_fmt=$(printf '%02d:%02d' $((exp_elapsed/60)) $((exp_elapsed%60)))
    echo "${config_name}|${clients}cl|${exp_fmt}" >> "$TIMES_FILE"
    COMPLETED=$((COMPLETED + 1))
    CURRENT_RUNNING=""
    CURRENT_CLIENTS=""

    echo "  >> Completato in $exp_fmt ($config_name $clients client)"
}

print_summary_table() {
    echo ""
    echo "========================================================================"
    echo "  TABELLA RIEPILOGATIVA TEMPI DI ESECUZIONE"
    echo "  +----------------------------+------------+-------------------+"
    echo "  | Esperimento                |  Client    |  Tempo            |"
    echo "  +----------------------------+------------+-------------------+"

    if [ -f "$TIMES_FILE" ]; then
        while IFS='|' read -r name clients time; do
            printf "  | %-26s | %-10s | %-17s |\n" "$name" "$clients" "$time"
        done < "$TIMES_FILE"
    fi

    echo "  +----------------------------+------------+-------------------+"
}

# ===================================================================
# INIZIO ESECUZIONE
# ===================================================================

echo ""
echo "========================================================================"
echo "  DHFLPL2 - Piano Sperimentale Completo"
echo "  Inizio: $START_TIME_FMT"
echo "  Totale: $TOTAL_TASKS task (50 exp + 1 plot + 6 demo + 1 summary)"
echo "========================================================================"

print_status_table

# ---------------------------------------------------------------
# FASE 1: ESPERIMENTI STANDARD (25)
# ---------------------------------------------------------------
CURRENT_PHASE=1
echo ""
echo "============================================================"
echo "  FASE 1/4: Esperimenti standard (senza DP) [25 esperimenti]"
echo "============================================================"

for config in cifar10 cifar100 mnist fashion_mnist svhn; do
    for clients in 2 5 10 20 50; do
        run_experiment "$config" "$config" "$clients"
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

for config in cifar10 cifar100 mnist fashion_mnist svhn; do
    dp_config="${config}_dp"
    for clients in 2 5 10 20 50; do
        run_experiment "$dp_config" "$dp_config" "$clients"
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

DEMO_MODULES="demos.demo_data_redaction demos.demo_dp_comparison demos.demo_gradient_inversion demos.demo_membership_inference demos.demo_model_update_leakage demos.demo_side_channel"
DEMO_NAMES="Data_Redaction DP_Comparison Gradient_Inversion Membership_Inference Model_Update_Leakage Side_Channel"
DEMO_ARGS="|--dataset mnist --rounds 30 --clients 2|--epsilon 1.0 --iterations 300|--rounds 15 --clients 2|--rounds 20 --clients 2|--clients 5 --rounds 10"

i=0
for module in $DEMO_MODULES; do
    i=$((i + 1))
    name=$(echo "$DEMO_NAMES" | cut -d' ' -f$i | tr '_' ' ')
    args=$(echo "$DEMO_ARGS" | cut -d'|' -f$((i+1)))
    TASK_NUM=$((TASK_NUM + 1))
    print_progress $TASK_NUM $TOTAL_TASKS "Demo: $name"

    local_start=$(date '+%s')
    python -m $module $args
    local_end=$(date '+%s')
    local_elapsed=$(( local_end - local_start ))
    local_fmt=$(printf '%02d:%02d' $((local_elapsed/60)) $((local_elapsed%60)))

    echo "Demo: $name||${local_fmt}" >> "$TIMES_FILE"
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

print_status_table
print_summary_table

echo ""
echo "========================================================================"
echo "  ESECUZIONE COMPLETATA"
echo "  Inizio:      $START_TIME_FMT"
echo "  Fine:        $END_TIME_FMT"
echo "  Durata:      $TOTAL_FMT"
echo "  Completati:  $COMPLETED task"
echo "  Saltati:     $SKIPPED task"
echo ""
echo "  Risultati:   experiments/results/*.json"
echo "  Grafici:     experiments/results/*.png"
echo "  Demo:        demos/outputs/*.png"
echo "  Log:         overnight_log.txt"
echo "========================================================================"
