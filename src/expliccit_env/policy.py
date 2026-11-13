"""The policy contract (sage-v1) between exported policies and the guest.

Every step (Sage's 10 ms control period, extended until an ACK arrives) the
guest builds Sage's 77-value state message for the observed flow, plus the
min RTT column expliCCit added (guest/src/step.rs, a port of Sage's sage.cc),
and runs the policy:

    input  "obs"   float32 [1, 78], the message, columns as in OBS_COLUMNS
    input  "state" float32 [1, S], optional; zeros at the first step
    input  "truth" float32 [1, W], optional, W at most TRUTH_DIM; the path's
                   true state at the step, the first W of TRUTH_COLUMNS (7
                   for a policy exported when the truth held seven)
    input  "cwnd"  float32 [1, 1], optional; the sender's window in packets
                   (TCP_INFO's snd_cwnd, the steps table's cwnd column)
    input  "cc_info" optional, from TCP_CC_INFO, the controller's statistics
                   since the previous step: float32 [1, 5] for mode "cubic"
                   (Orca's: RTT samples, mean rate sample in bytes/s, lost
                   bytes, the period in us, min RTT in us), float32 [1, 7] for
                   mode "rate" (Aurora's monitor interval: its length in us,
                   packets sent, delivered and lost, RTT samples, their mean
                   in us, the second half's mean less the first's in us)
    output 0       float32 [1, 1], a: the window is multiplied by 2**a
    output 1       float32 [1, S], the next state, present iff "state" is

Inputs are matched by name, in any order. "truth" is what an oracle policy
(net2 on the true LCFs) or a patched one (some of net1's beliefs replaced by
the truth) reads: the guest computes it live from the bottleneck's records,
with the same definitions as the labels (labels.labels_at), so it equals the
labels its step is stored with. Its values can be NaN where a label is
(mean queueing delay with no departure in the window, free buffer without a
packet limit, busy fraction at zero capacity); the graph must handle that.
"cwnd" lets a policy choose an absolute window and still answer with a
multiplier: a = log2(target) - log2(cwnd).

The policy chooses its own columns and does its own scaling inside the graph,
so any network design fits without changing the guest. The message is Sage's
as is, so a column can be infinite or NaN where sage.cc divides by a zero
(for example column 68 before any sending rate is known); a policy must not
let such a column reach its output. |a| must be at most
MAX_ACTION and finite; anything else fails the episode instead of being
clamped, so a broken policy cannot silently produce data. How a sets the
window depends on the agent's mode (spec.AgentCC): in "hold" and "pure" it
follows Sage, a precise accumulator from 10 packets, rounded up, kept in
[4, 20000] (pacing cwnd * mss / srtt once a step in "hold", Sage's kernel
pacing in "pure"); in "cubic" the window is the kernel's current one times
2**a, rounded down, in [4, 20000], with Cubic in between; in "rate" a
multiplies a sending rate (from 6 Mbit/s, in [0.5, 300] Mbit/s) the kernel
paces at, and each step lasts half the last step's mean RTT. An agent may run
its policy at a longer period than Sage's 10 ms (AgentCC.period_ms).
"""

from __future__ import annotations

from typing import Final

from expliccit_env.labels import WINDOWED, WINDOWS_MS, label_name

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
# The "truth" input: these labels (labels.labels_at), in their units, which
# the guest computes live (guest/src/truth.rs). It is every stored label
# known at the step, so all but queue_delay_ns, the delay of a packet that
# may not have left yet. expliCCit's seven come first: a policy whose truth
# input is narrower reads a prefix. Then the other instant labels, the other
# windowed labels over the first window, and every windowed label over the
# other windows.
_SEVEN: Final = (
    "queue_delay_mean_ns",
    "queue_pkts",
    "capacity_bps",
    "free_buffer_pkts",
    "busy_fraction",
    "one_way_delay_ns",
    "receive_rate_bps",
)
TRUTH_COLUMNS: Final = (
    *_SEVEN,
    "queue_bytes",
    "flow_queue_pkts",
    "buffer_fill",
    "base_rtt_ns",
    "buffer_pkts",
    "buffer_bytes",
    "random_loss_rate",
    *(name for name in WINDOWED if name not in _SEVEN),
    *(label_name(name, w) for w in WINDOWS_MS[1:] for name in WINDOWED),
)
TRUTH_DIM: Final = 36

assert len(OBS_COLUMNS) == OBS_DIM and len(SAGE_OBS_COLS) == 69
assert len(EXPLICCIT_OBS_COLS) == 70
assert len(TRUTH_COLUMNS) == TRUTH_DIM
