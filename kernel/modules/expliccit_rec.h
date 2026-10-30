/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Record ABI shared by the expliccit-env kernel modules and the host decoder
 * (src/expliccit_env/records.py, which mirrors these layouts and is tested
 * against them). Records are little-endian, a multiple of 8 bytes long, and
 * carry their own length. Any layout change bumps EXPL_REC_ABI.
 *
 * Time fields: t_ns is CLOCK_MONOTONIC (ktime_get_ns) of the event itself,
 * which is not always the moment the record was emitted; each record type
 * below says which instant it means. ms fields are link-model milliseconds
 * since the qdisc's base_ns.
 */
#ifndef _EXPLICCIT_REC_H
#define _EXPLICCIT_REC_H

#include <linux/types.h>

#define EXPL_REC_ABI 3
#define EXPL_REC_MAGIC 0x52504c45 /* "ELPR" little-endian */

enum expl_rec_type {
	EXPL_REC_META = 1,
	EXPL_REC_LINK_ARRIVAL = 2,
	EXPL_REC_LINK_DROP = 3,
	EXPL_REC_LINK_DEPART = 4,
	EXPL_REC_LINK_OPP = 5,
	EXPL_REC_LINK_STATS = 6,
	EXPL_REC_TCP_OBS = 7,
	EXPL_REC_AGENT_ACTION = 8,
};

struct expl_rec_hdr {
	__u16 type;	/* enum expl_rec_type */
	__u16 len;	/* bytes including this header */
	__u32 seq;	/* emission counter; a gap means the ring overflowed */
	__u64 t_ns;
};

/* First record a reader sees after open(); t_ns is the open time. */
struct expl_rec_meta {
	struct expl_rec_hdr hdr;
	__u32 magic;
	__u32 abi;
	__u32 hz;
	__u32 ring_bytes;
};

enum expl_drop_reason {
	EXPL_DROP_NONE = 0,
	EXPL_DROP_QUEUE_FULL = 1,	/* droptail, Mahimahi's "d" event */
	EXPL_DROP_DELAY_FULL = 2,	/* delay-line safety bound: invalid episode */
	/* Not a well-formed IPv4 packet of at most 1500 bytes: invalid episode. */
	EXPL_DROP_OVERSIZE = 3,
	EXPL_DROP_GSO = 4,		/* GSO segmentation failed: invalid episode */
	EXPL_DROP_RANDOM = 5,		/* the i.i.d. loss stage (Mahimahi mm-loss) */
};

/*
 * LINK_ARRIVAL: a packet reached the link stage (Mahimahi "+"). Emitted for
 *   every packet, including ones then dropped. t_ns is the exact arrival;
 *   ms = arrival_ms = its millisecond. qlen_* is the queue the packet found.
 * LINK_DROP: the packet just announced by LINK_ARRIVAL was dropped, or a
 *   packet failed before reaching the link (reason says which).
 * LINK_DEPART: the packet's last byte left on the opportunity at ms
 *   (Mahimahi "-"; queueing delay is ms - arrival_ms). t_ns is that
 *   opportunity's instant. qlen_* is the queue left behind.
 * qlen_* never counts the packet in transit, matching Mahimahi's droptail.
 * bytes is the IPv4 total length plus Mahimahi's 4-byte TUN header.
 */
struct expl_rec_link {
	struct expl_rec_hdr hdr;
	__u32 link_id;
	__u32 bytes;
	__u64 ms;
	__u64 arrival_ms;
	__u32 qlen_pkts;
	__u32 qlen_bytes;
	__u32 tcp_seq;
	__u32 tcp_ack;
	__u16 sport;
	__u16 dport;
	__u8 proto;
	__u8 tcp_flags;
	__u8 reason;		/* enum expl_drop_reason, NONE unless LINK_DROP */
	__u8 pad;
};

/*
 * LINK_OPP: all `count` delivery opportunities (1504 bytes each) at ms were
 *   processed; used_bytes of them carried packet data (Mahimahi "#" lines,
 *   one record per millisecond). The qdisc processes opportunities lazily, so
 *   a millisecond with no queued packet may have no record; the trace itself
 *   is the authority on capacity. t_ns is the millisecond's start instant.
 */
struct expl_rec_opp {
	struct expl_rec_hdr hdr;
	__u32 link_id;
	__u32 count;
	__u64 ms;
	__u32 used_bytes;
	__u32 pad;
};

/* LINK_STATS: emitted once when the qdisc is destroyed; t_ns is that time.
 * Any nonzero drops_delay, drops_oversize or drops_gso invalidates the
 * episode; drops_random counts the loss stage's intended drops. purged
 * counts packets still held when the qdisc was reset.
 */
struct expl_rec_link_stats {
	struct expl_rec_hdr hdr;
	__u32 link_id;
	__u32 pad;
	__u64 arrivals;
	__u64 departures;
	__u64 drops_queue;
	__u64 drops_delay;
	__u64 drops_oversize;
	__u64 drops_gso;
	__u64 bypassed;
	__u64 early;
	__u64 purged;
	__u64 drops_random;
};

#define EXPL_OBS_RATE_APP_LIMITED	0x1
#define EXPL_OBS_CWND_LIMITED		0x2

/*
 * TCP_OBS: sender state sampled when an incoming segment (normally an ACK)
 *   enters tcp_rcv_established(), BEFORE that segment is processed. So record
 *   k holds the state left by segment k-1 plus any timer-driven changes since.
 *   ack_seq, data_len and tcp_flags describe segment k itself. t_ns is the
 *   sample instant. Units: *_us microseconds, pacing_rate bytes per second,
 *   delivered_mstamp microseconds (tcp_clock_us).
 */
struct expl_rec_tcp_obs {
	struct expl_rec_hdr hdr;
	__u16 sport;
	__u16 dport;
	__u8 ca_state;
	__u8 tcp_flags;
	__u16 data_len;
	__u32 ack_seq;
	__u32 snd_una;
	__u32 snd_nxt;
	__u32 snd_cwnd;
	__u32 snd_ssthresh;
	__u32 srtt_us;
	__u32 mdev_us;
	__u32 min_rtt_us;
	__u32 last_rtt_us;
	__u32 packets_out;
	__u32 sacked_out;
	__u32 lost_out;
	__u32 retrans_out;
	__u32 delivered;
	__u32 delivered_ce;
	__u32 lost;
	__u32 total_retrans;
	__u32 rate_delivered;
	__u32 rate_interval_us;
	__u32 app_limited;
	__u32 mss_cache;
	__u32 snd_wnd;
	__u32 notsent_bytes;
	__u32 flags;		/* EXPL_OBS_* */
	__u64 bytes_acked;
	__u64 bytes_sent;
	__u64 bytes_retrans;
	__u64 pacing_rate;
	__u64 delivered_mstamp;
};

/*
 * AGENT_ACTION: tcp_agent accepted an action written to /dev/expl_agent
 *   (struct expl_agent_action). t_ns is the write time; the action takes
 *   effect at each agent socket's next ACK. action_seq counts accepted
 *   actions from 1.
 */
struct expl_rec_agent_action {
	struct expl_rec_hdr hdr;
	__u32 cwnd;
	__u32 action_seq;
	__u64 pacing_rate;
};

/*
 * What a policy writes to /dev/expl_agent: exactly one struct per write().
 * cwnd is in packets, 2..EXPL_AGENT_MAX_CWND; pacing_rate is in bytes per
 * second, 0 meaning unpaced, at most EXPL_AGENT_MAX_PACING.
 */
#define EXPL_AGENT_ABI 1
#define EXPL_AGENT_MAX_CWND 1048576U		/* 2^20 */
#define EXPL_AGENT_MAX_PACING 1099511627776ULL	/* 2^40 */

struct expl_agent_action {
	__u32 abi;
	__u32 cwnd;
	__u64 pacing_rate;
};

#endif /* _EXPLICCIT_REC_H */
