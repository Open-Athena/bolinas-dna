#!/bin/bash
# Fast-cycling launcher for Lambda GPUs. Sky's --retry-until-up pins to
# the first region in the failover plan and won't actually cycle regions
# (verified empirically — kept hitting us-east-1 in a loop). Lambda's
# capacity comes and goes within seconds; this loop tries each region
# with a short single-shot launch so we land on whichever pops up.
#
# Stops on first success (cluster status UP). Caps at MAX_ROUNDS to avoid
# runaway. After each completed round of all regions, sleeps briefly.
#
# Usage: ./sky_cycle_gh200.sh CLUSTER_NAME YAML_PATH [extra_sky_args...]
# The GPU type is passed via the extra args (--gpus GH200:1 or H100:1).

set -uo pipefail

CLUSTER="${1:?need cluster name}"
YAML="${2:?need yaml path}"
shift 2

REGIONS=(
  us-east-1 us-east-2 us-east-3
  us-west-1 us-west-2 us-west-3
  us-south-1 us-south-2 us-south-3
  us-midwest-1
  europe-central-1 europe-south-1
  me-west-1
  asia-south-1 asia-northeast-1 asia-northeast-2
  australia-east-1
)

MAX_ROUNDS="${MAX_ROUNDS:-20}"
ROUND_SLEEP="${ROUND_SLEEP:-15}"

round=0
while (( round < MAX_ROUNDS )); do
  round=$((round + 1))
  echo "===== round $round/$MAX_ROUNDS ====="
  for region in "${REGIONS[@]}"; do
    echo "--- $region ---"
    if sky launch -y -c "$CLUSTER" --infra "lambda/$region" "$YAML" "$@" 2>&1 \
        | grep -E -i "Launching on|capacity|provisioned|Cluster.* is up|^E |Setup completed|Job submitted|cluster up|getcwd"; then
      :
    fi
    # Exit code of sky comes from the leftmost command in the pipe; check separately.
    if sky status -v 2>/dev/null | grep -E "^${CLUSTER}\s.*UP\s" >/dev/null; then
      echo ">>> Cluster ${CLUSTER} is UP in $region — exiting loop."
      exit 0
    fi
  done
  echo "===== round $round done, sleeping ${ROUND_SLEEP}s ====="
  sleep "$ROUND_SLEEP"
done

echo "Exhausted $MAX_ROUNDS rounds. Last cluster state:"
sky status -v 2>&1 | tail -5
exit 1
