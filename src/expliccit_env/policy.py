"""The policy contract (tcpinfo-v1) between exported policies and the guest.

An ONNX policy has one float32 input of shape [1, 8] and one float32 output
of shape [1, 2]. The guest (guest/src/agent.rs) fills the input from
TCP_INFO every ``interval_ms`` of virtual time and applies the output
through tcp_agent. Out-of-range outputs fail the episode instead of being
clamped, so a broken policy cannot silently produce data.
"""

from __future__ import annotations

from typing import Final

FEATURES: Final = (
    "srtt_ms",
    "min_rtt_ms",
    "delivery_rate_mbps",
    "cwnd_pkts",
    "unacked_pkts",
    "lost_pkts",
    "retrans_in_interval",
    "acked_mbps_in_interval",
)
ACTIONS: Final = ("cwnd_pkts", "pacing_mbps")  # pacing 0 = unpaced
CWND_RANGE: Final = (2, 2**20)
