#!/usr/bin/env bash
# Host side of the M0 network probe.
# Usage: run-net.sh UML_BINARY OUT_DIR TAG CC RATE DELAY SECS [MODULE.ko]
set -euo pipefail
uml="${1:?}"; out="${2:?}"; tag="${3:?}"; cc="${4:?}"; rate="${5:?}"; delay="${6:?}"; secs="${7:?}"
mod="${8:-}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
extra=()
[[ -n "$mod" ]] && extra+=("probe_mod=$(realpath "$mod")")
start="$(date +%s.%N)"
rc=0
timeout 900 "$uml" mem=512M time-travel=inf-cpu rootfstype=hostfs rootflags=/ ro \
    init="$here/guest-net.sh" con=null con0=null,fd:1 \
    probe_cc="$cc" probe_rate="$rate" probe_delay="$delay" probe_secs="$secs" \
    probe_out="$out" probe_tag="$tag" "${extra[@]}" > "$out/boot-$tag.log" 2>&1 || rc=$?
end="$(date +%s.%N)"
echo "HOST tag=$tag cc=$cc rate=$rate delay=$delay secs=$secs rc=$rc wall=$(echo "$end - $start" | bc)"
grep -E 'virtual_secs|tcp_probe_hits|insmod' "$out/net-$tag.txt" || true
python3 -c 'import json,sys; e=json.load(open(sys.argv[1]))["end"]; print("throughput_mbps", e["sum_received"]["bits_per_second"]/1e6, "retrans", e["sum_sent"].get("retransmits"))' "$out/iperf-$tag.json"
