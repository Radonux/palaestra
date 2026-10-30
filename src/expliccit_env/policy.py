"""The policy contract (sage-v1) between exported policies and the guest.

Every step (Sage's 10 ms control period, extended until an ACK arrives) the
guest builds Sage's 77-value state message for the observed flow, plus the
min RTT column expliCCit added (guest/src/step.rs, a port of Sage's sage.cc),
and runs the policy:

    input  "obs"   float32 [1, 78], the message, columns as in OBS_COLUMNS
    input  "state" float32 [1, S], optional; zeros at the first step
    output 0       float32 [1, 1], a: the window is multiplied by 2**a
    output 1       float32 [1, S], the next state, present iff "state" is

The policy chooses its own columns and does its own scaling inside the graph,
so any network design fits without changing the guest. The message is Sage's
as is, so a column can be infinite or NaN where sage.cc divides by a zero
(for example column 68 before any sending rate is known); a policy must not
let such a column reach its output. |a| must be at most
MAX_ACTION and finite; anything else fails the episode instead of being
clamped, so a broken policy cannot silently produce data. The window follows
Sage: a precise accumulator from 10 packets, rounded up, kept in [4, 20000];
pacing is cwnd * mss / srtt.
"""

from __future__ import annotations

from typing import Final

_W = tuple(f"{w}_{s}" for w in ("short", "mid", "long") for s in ("avg", "min", "max"))

# Sage's message, in sage.cc's order. Units follow Sage: "x100ms" values are
# in units of 100 ms, "x100mbps" in units of 100 Mbit/s.
OBS_COLUMNS: Final = (
    "t_s",  # time on the trace, s
    "unused_capacity",  # Sage: the link's nominal capacity; always 0 here
    "srtt_x100ms",
    "rttvar_ms",
    "rto_x100ms",
    "ato_x100ms",
    "pacing_rate_x100mbps",
    "delivery_rate_x100mbps",
    "ssthresh_pkts",
    "ca_state",
    *(f"srtt_x100ms_{w}" for w in _W),
    *(f"delivery_rate_x100mbps_{w}" for w in _W),
    *(f"min_rtt_over_srtt_{w}" for w in _W),
    *(f"rttvar_ms_{w}" for w in _W),
    *(f"unacked_kpkts_{w}" for w in _W),
    *(f"lost_x100pkts_{w}" for w in _W),
    "delivered_minus_lost_mbps",  # over the last 100 steps
    "step_over_min_rtt",
    "min_rtt_over_srtt",
    "lost_x100mbps",
    "acked_over_max_sent",
    "delivered_ratio",  # this step's windowed rate over the previous step's
    "max_delivered_bdp_over_cwnd",
    "delivered_x100mbps",
    "unacked_over_sent",
    "max_delivered_ratio",
    "max_delivered_x100mbps",
    "unused_reward",  # Sage's reward; always 0 here, the host computes it
    "prev_action",  # log2 of the window multiplier since the last step
    "min_rtt_ms",  # added by expliCCit's sage.cc (there at index 75)
)
OBS_DIM: Final = 78
# Sage's own observation (rl_module/config-rl-eval.yaml, obs_cols, 69 columns).
SAGE_OBS_COLS: Final = (
    2, 3, 7, 9, *range(10, 64), *range(65, 75), 76,
)  # fmt: skip
# expliCCit's observation (its config-rl-eval.yaml): Sage's, then min RTT, then
# the previous action, in this order, in this message's indices.
EXPLICCIT_OBS_COLS: Final = (*SAGE_OBS_COLS[:-1], 77, 76)
PREV_ACTION_COL: Final = 76
MAX_ACTION: Final = 8.0

assert len(OBS_COLUMNS) == OBS_DIM and len(SAGE_OBS_COLS) == 69
assert len(EXPLICCIT_OBS_COLS) == 70
