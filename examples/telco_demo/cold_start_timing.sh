#!/bin/bash
# Measure cold-start time for each vLLM lane.
# Stops the container, waits 2s, then docker start + polls /health every 1s
# until it returns 200. Records wall time.
set -e

declare -A PORTS=(
  [vllm-economy]=8001
  [vllm-standard]=8002
  [vllm-premium]=8003
)

declare -A LABELS=(
  [vllm-economy]="Qwen2.5-3B / 1xA100 / 4k ctx / 6GB weights"
  [vllm-standard]="Qwen2.5-14B / 1xA100 / 16k ctx / 28GB weights"
  [vllm-premium]="Qwen2.5-72B / 4xA100 TP=4 / 32k ctx / 144GB weights"
)

results_file=/tmp/cold_start_results.txt
> "$results_file"

for container in vllm-economy vllm-standard vllm-premium; do
  port=${PORTS[$container]}
  label=${LABELS[$container]}
  echo
  echo "================================================================"
  echo "  cold-start test:  $container"
  echo "  $label"
  echo "================================================================"

  echo "  [1] stopping container..."
  t_stop_start=$(date +%s)
  docker stop "$container" >/dev/null
  t_stop_end=$(date +%s)
  stop_dur=$((t_stop_end - t_stop_start))
  echo "      stopped in ${stop_dur}s"

  sleep 2

  echo "  [2] starting container..."
  t_start=$(date +%s)
  docker start "$container" >/dev/null

  echo "  [3] polling http://localhost:${port}/health..."
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    if curl -sf -m 2 "http://localhost:${port}/health" >/dev/null 2>&1; then
      t_ready=$(date +%s)
      total=$((t_ready - t_start))
      echo "      HEALTHY at attempt ${attempt}, total ${total}s"
      echo "${container}: stop=${stop_dur}s start_to_healthy=${total}s   ${label}" >> "$results_file"
      break
    fi
    if [ $attempt -gt 600 ]; then
      echo "      TIMEOUT after 600s"
      echo "${container}: TIMEOUT after 600s   ${label}" >> "$results_file"
      break
    fi
    sleep 1
  done

  # quick smoke that it's actually serving
  echo "  [4] smoke test: 1 generation request..."
  smoke=$(curl -s -X POST "http://localhost:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
    -w 'HTTP %{http_code} in %{time_total}s' -o /dev/null)
  echo "      $smoke"

  sleep 3  # let TokenFlow re-probe
done

echo
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
cat "$results_file"
