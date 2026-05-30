#!/usr/bin/env bash

set -euo pipefail

cd /workspace/isaaclab

run_benchmark() {
    local task="$1"
    local num_envs="$2"
    local steps="$3"
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
        --steps "$steps" \
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

benchmark_task() {
    local task="$1"
    local steps="$2"
    local results_file="$3"
    shift 3

    local num_envs_pow
    local num_envs
    local total_time

    for num_envs_pow in "$@"; do
        num_envs=$((2 ** num_envs_pow))
        echo "Benchmarking $task with $num_envs envs..."
        total_time="$(run_benchmark "$task" "$num_envs" "$steps")"

        if [[ "$total_time" == "-1" ]]; then
            echo "Steps/s for $num_envs envs: -1"
            printf '%s\t%s\t%s\n' "$task" "$num_envs" "-1" >> "$results_file"
            break
        fi

        local steps_per_second
        steps_per_second="$(awk -v steps="$steps" -v total_time="$total_time" 'BEGIN { if (total_time > 0) print steps / total_time; else print -1 }')"
        echo "Steps/s for $num_envs envs: $steps_per_second"
        printf '%s\t%s\t%s\n' "$task" "$num_envs" "$steps_per_second" >> "$results_file"
        sleep 5
    done
}

benchmark_multiple_tasks() {
    local steps="$1"
    local results_file="$2"
    shift
    shift

    local task
    local task_spec
    local exponents

    for task_spec in "$@"; do
        IFS='|' read -r task exponents <<< "$task_spec"
        echo "Benchmarking task $task..."
        # shellcheck disable=SC2206
        benchmark_task "$task" "$steps" "$results_file" ${exponents}
    done
}

main() {
    local steps=1000
    local results_file
    local final_results

    results_file="$(mktemp)"

    benchmark_multiple_tasks \
        "$steps" \
        "$results_file" \
        "Isaac-Cartpole-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
        "Isaac-Ant-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
        "Isaac-Repose-Cube-Shadow-Direct-v0|0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

    final_results="$(/workspace/isaaclab/isaaclab.sh -p - "$results_file" <<'PY'
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
)"

    rm -f "$results_file"

    echo "Final results:"
    echo "$final_results"
}

main "$@"
