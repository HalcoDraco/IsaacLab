#!/usr/bin/env bash

set -euo pipefail

cd /workspace/isaaclab

OUTPUT_FILE="/workspace/isaaclab/scripts/server_mode/benchmark_non_server_mode_results.txt"

run_benchmark() {
    local task="$1"
    local num_envs="$2"
    local result_file
    local log_file
    local exit_code

    result_file="$(mktemp)"
    log_file="$(mktemp)"

    set +e
    ./isaaclab.sh -p scripts/server_mode/env_loop_non_server_mode.py \
        --headless \
        --task "$task" \
        --num_envs "$num_envs" \
        --result_file "$result_file" \
        >"$log_file" 2>&1
    exit_code=$?
    set -e

    cat "$log_file" >&2

    if [[ "$exit_code" -ne 0 ]]; then
        echo "Benchmark command failed for task $task with $num_envs envs" >&2
        rm -f "$result_file" "$log_file"
        return "$exit_code"
    fi

    if [[ ! -s "$result_file" ]]; then
        echo "Failed to read total time for task $task with $num_envs envs" >&2
        rm -f "$result_file" "$log_file"
        return 1
    fi

    tr -d '[:space:]' < "$result_file"

    rm -f "$result_file" "$log_file"
}

format_results() {
    local results_file="$1"

    /workspace/isaaclab/isaaclab.sh -p - "$results_file" <<'PY'
import sys
from collections import defaultdict

results_path = sys.argv[1]
results = defaultdict(dict)

with open(results_path, encoding='utf-8') as file_handle:
    for line in file_handle:
        task, num_envs, steps_per_second = line.rstrip('\n').split('\t')
        results[task][int(num_envs)] = float(steps_per_second)

print(dict(results))
PY
}

benchmark_task() {
    local task="$1"
    local results_file="$2"
    shift 2

    local num_envs_pow
    local num_envs
    local total_time
    local run
    local runs=3
    local sum_time
    local avg_time
    local valid_runs
    local partial_results

    local STEPS=10000
    for num_envs_pow in "$@"; do
        num_envs=$((2 ** num_envs_pow))
        echo "Benchmarking $task with $num_envs envs..."
        sum_time=0
        valid_runs=0

        for run in $(seq 1 "$runs"); do
            echo "Run $run/$runs for $num_envs envs..."
            total_time="$(run_benchmark "$task" "$num_envs")"

            if [[ "$total_time" == "-1" ]]; then
                echo "Run $run failed."
                break
            fi

            echo "Steps/s for run $run: $(awk -v steps="$STEPS" -v total_time="$total_time" 'BEGIN { print steps / total_time }')"
            sum_time="$(awk -v sum="$sum_time" -v total_time="$total_time" 'BEGIN { print sum + total_time }')"
            valid_runs=$((valid_runs + 1))
            sleep 8
        done

        if [[ "$valid_runs" -eq "$runs" ]]; then
            avg_time="$(awk -v sum="$sum_time" -v runs="$runs" 'BEGIN { print sum / runs }')"
            local steps_per_second
            steps_per_second="$(awk -v steps="$STEPS" -v total_time="$avg_time" 'BEGIN { if (total_time > 0) print steps / total_time; else print -1 }')"
            echo "Average Steps/s for $num_envs envs: $steps_per_second"
            printf '%s\t%s\t%s\n' "$task" "$num_envs" "$steps_per_second" >> "$results_file"
        else
            echo "Steps/s for $num_envs envs: -1"
            printf '%s\t%s\t%s\n' "$task" "$num_envs" "-1" >> "$results_file"
        fi

        # Persist partial results after each iteration so they survive a crash
        # (best-effort: a conversion hiccup must not abort the benchmark run).
        if partial_results="$(format_results "$results_file")"; then
            printf '%s\n' "$partial_results" > "$OUTPUT_FILE"
        else
            echo "Warning: failed to persist partial results to $OUTPUT_FILE" >&2
        fi

        if [[ "$valid_runs" -ne "$runs" ]]; then
            break
        fi
    done
}

benchmark_multiple_tasks() {
    local results_file="$1"
    shift

    local task
    local task_spec
    local exponents

    for task_spec in "$@"; do
        IFS='|' read -r task exponents <<< "$task_spec"
        echo "Benchmarking task $task..."
        # shellcheck disable=SC2206
        benchmark_task "$task" "$results_file" ${exponents}
    done
}

main() {
    local results_file
    local final_results

    results_file="$(mktemp)"

    benchmark_multiple_tasks \
        "$results_file" \
        "Isaac-Cartpole-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
        "Isaac-Ant-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
        "Isaac-Repose-Cube-Shadow-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

    final_results="$(format_results "$results_file")"

    rm -f "$results_file"

    echo "Final results:"
    echo "$final_results"
    printf '%s\n' "$final_results" > "$OUTPUT_FILE"
    echo "Results saved to $OUTPUT_FILE"
}

main "$@"
