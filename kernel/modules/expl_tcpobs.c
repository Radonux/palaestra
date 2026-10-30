// SPDX-License-Identifier: GPL-2.0
/*
 * expl_tcpobs: sender-side observations for expliCCit's net1.
 *
 * Attaches to the tcp:tcp_probe tracepoint (found by name, it is not exported)
 * and, for every segment arriving on a socket whose SO_MARK equals the "mark"
 * parameter, emits one EXPL_REC_TCP_OBS record with the socket state
 * immediately before the segment is processed (see expliccit_rec.h). Works
 * for any congestion control, because it reads only the generic tcp_sock.
 * kprobes are unavailable in UML, which is why this is a tracepoint.
 *
 * Context: tcp_rcv_established(), i.e. softirq or socket-backlog processing,
 * always with the socket locked, so plain reads of tcp_sock are consistent.
 */
#include <linux/module.h>
#include <linux/string.h>
#include <linux/tracepoint.h>
#include <net/inet_connection_sock.h>
#include <net/tcp.h>

#include "expl_rec.h"

static unsigned int mark;
module_param(mark, uint, 0644);
MODULE_PARM_DESC(mark, "observe sockets with this SO_MARK; 0 observes none");

static struct tracepoint *tp_tcp_probe;

static void expl_tcpobs_probe(void *data, struct sock *sk,
			      const struct sk_buff *skb)
{
	const struct tcphdr *th = (const struct tcphdr *)skb->data;
	const struct tcp_sock *tp = tcp_sk(sk);
	const struct inet_sock *inet = inet_sk(sk);
	u32 want = READ_ONCE(mark);
	struct expl_rec_tcp_obs r;

	if (!want || READ_ONCE(sk->sk_mark) != want)
		return;

	r = (struct expl_rec_tcp_obs) {
		.hdr = { .type = EXPL_REC_TCP_OBS, .len = sizeof(r),
			 .t_ns = ktime_get_ns() },
		.sport = ntohs(inet->inet_sport),
		.dport = ntohs(inet->inet_dport),
		.ca_state = inet_csk(sk)->icsk_ca_state,
		.tcp_flags = tcp_flag_byte(th),
		.data_len = skb->len - __tcp_hdrlen(th),
		.ack_seq = ntohl(th->ack_seq),
		.snd_una = tp->snd_una,
		.snd_nxt = tp->snd_nxt,
		.snd_cwnd = tcp_snd_cwnd(tp),
		.snd_ssthresh = tp->snd_ssthresh,
		.srtt_us = tp->srtt_us >> 3,
		.mdev_us = tp->mdev_us >> 2,
		.min_rtt_us = tcp_min_rtt(tp),
		.last_rtt_us = tp->rack.rtt_us,
		.packets_out = tp->packets_out,
		.sacked_out = tp->sacked_out,
		.lost_out = tp->lost_out,
		.retrans_out = tp->retrans_out,
		.delivered = tp->delivered,
		.delivered_ce = tp->delivered_ce,
		.lost = tp->lost,
		.total_retrans = tp->total_retrans,
		.rate_delivered = tp->rate_delivered,
		.rate_interval_us = tp->rate_interval_us,
		.app_limited = tp->app_limited,
		.mss_cache = tp->mss_cache,
		.snd_wnd = tp->snd_wnd,
		.notsent_bytes = READ_ONCE(tp->write_seq) - tp->snd_nxt,
		.flags = (tp->rate_app_limited ? EXPL_OBS_RATE_APP_LIMITED : 0) |
			 (tp->is_cwnd_limited ? EXPL_OBS_CWND_LIMITED : 0),
		.bytes_acked = tp->bytes_acked,
		.bytes_sent = tp->bytes_sent,
		.bytes_retrans = tp->bytes_retrans,
		.pacing_rate = READ_ONCE(sk->sk_pacing_rate),
		.delivered_mstamp = tp->delivered_mstamp,
	};
	expl_rec_emit(&r.hdr);
}

static void find_tcp_probe(struct tracepoint *tp, void *priv)
{
	if (!strcmp(tp->name, "tcp_probe"))
		*(struct tracepoint **)priv = tp;
}

static int __init expl_tcpobs_init(void)
{
	for_each_kernel_tracepoint(find_tcp_probe, &tp_tcp_probe);
	if (!tp_tcp_probe)
		return -ENOENT;
	return tracepoint_probe_register(tp_tcp_probe, expl_tcpobs_probe, NULL);
}

static void __exit expl_tcpobs_exit(void)
{
	tracepoint_probe_unregister(tp_tcp_probe, expl_tcpobs_probe, NULL);
	tracepoint_synchronize_unregister();
}

module_init(expl_tcpobs_init);
module_exit(expl_tcpobs_exit);
MODULE_DESCRIPTION("expliccit-env sender observations from tcp_probe");
MODULE_LICENSE("GPL");
