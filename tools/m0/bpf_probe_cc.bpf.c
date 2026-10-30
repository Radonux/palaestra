// SPDX-License-Identifier: GPL-2.0
/* M0 probe: the smallest tcp_congestion_ops struct_ops map that libbpf will
 * try to register. Registering needs an arch BPF trampoline (decision D5).
 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

SEC("struct_ops")
__u32 BPF_PROG(m0_ssthresh, struct sock *sk)
{
	return 2;
}

SEC("struct_ops")
void BPF_PROG(m0_cong_avoid, struct sock *sk, __u32 ack, __u32 acked)
{
}

SEC("struct_ops")
__u32 BPF_PROG(m0_undo_cwnd, struct sock *sk)
{
	return 10;
}

SEC(".struct_ops.link")
struct tcp_congestion_ops m0_cc = {
	.ssthresh = (void *)m0_ssthresh,
	.cong_avoid = (void *)m0_cong_avoid,
	.undo_cwnd = (void *)m0_undo_cwnd,
	.name = "m0_bpf_cc",
};
