// SPDX-License-Identifier: GPL-2.0
/*
 * sch_mmlink: a trace-driven bottleneck with Mahimahi semantics.
 *
 * One qdisc is one direction of "mm-delay D mm-link TRACE" with a droptail
 * queue. Two root qdiscs on the path namespace's egress devices reproduce
 * Mahimahi's shell nesting:
 *
 *   data (sender to receiver): delay line, then link (ORDER_DELAY_LINK)
 *   ACKs (receiver to sender): link, then delay line (ORDER_LINK_DELAY)
 *
 * The link is a port of mahimahi src/frontend/link_queue.cc (LinkQueue):
 *   - time is integer milliseconds since base_ns; trace entry i holds the
 *     number of 1504-byte delivery opportunities at ms i;
 *   - the trace repeats with period equal to its last index, so the last ms
 *     of one cycle and ms 0 of the next coincide (as in Mahimahi);
 *   - before a packet arriving in ms a joins the queue, every opportunity at
 *     ms <= a is used, so a packet never departs in its arrival millisecond;
 *   - the queue head moves "in transit" and may span opportunities; it
 *     departs at the opportunity carrying its last byte;
 *   - packet size is the IPv4 total length plus Mahimahi's 4-byte TUN header;
 *   - droptail limits apply to queued packets, never to the one in transit.
 * The delay line is mahimahi's DelayQueue: release exactly delay_ns later. A
 * packet leaving the link at opportunity ms t enters the delay line at the
 * instant of t, so both directions share one millisecond grid.
 *
 * GSO packets are segmented on entry (as tbf does), because Mahimahi only
 * ever sees MTU-sized packets. Non-IPv4 frames (ARP) bypass the model, since
 * Mahimahi works on IP packets and never sees them.
 *
 * An optional i.i.d. loss stage (Mahimahi's mm-loss) drops packets on
 * entry, before the delay line and the link.
 *
 * Every arrival, drop, departure and processed opportunity is reported to
 * expl_rec (see expliccit_rec.h for what each record means).
 *
 * Locking: enqueue, dequeue, reset and the watchdog-driven dequeue run under
 * the qdisc root lock, in softirq or in process context with BH disabled.
 * init and destroy run under RTNL.
 *
 * Cost: O(1) per packet event plus O(1) per elapsed trace millisecond, paid
 * lazily when the qdisc next runs.
 */
#include <linux/ip.h>
#include <linux/module.h>
#include <linux/random.h>
#include <linux/skbuff.h>
#include <linux/slab.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <net/gso.h>
#include <net/netlink.h>
#include <net/pkt_sched.h>
#include <net/sch_generic.h>

#include "expl_rec.h"
#include "sch_mmlink.h"

#define MMLINK_PACKET_SIZE 1504	/* mahimahi LinkQueue::PACKET_SIZE */
#define MMLINK_TUN_HDR 4	/* tun_pi header Mahimahi counts per packet */
#define MMLINK_DELAY_LIMIT_DEFAULT BIT(20)

/* Per-packet state in skb->cb. t is a release time in ns while the packet
 * sits in the delay line and its arrival ms while it sits in the link queue.
 */
struct mmlink_cb {
	u64 t;
	u32 len;
};

struct mmlink_stats {
	u64 arrivals;
	u64 departures;
	u64 drops_queue;
	u64 drops_delay;
	u64 drops_oversize;
	u64 drops_gso;
	u64 bypassed;
	u64 early;
	u64 purged;
	u64 drops_random;
};

struct mmlink_sched {
	/* Configuration, fixed after init. */
	u64 base_ns;
	u64 delay_ns;
	u8 *trace;		/* owned; trace[period] != 0 */
	u32 period;
	u32 limit_pkts;
	u32 limit_bytes;
	u32 delay_limit;
	u32 link_id;
	u32 loss;		/* drop probability times 2^32 */
	enum mmlink_order order;
	bool configured;

	/* Next opportunity: ms cycle_ms + idx; all of ms < that were used. */
	u64 cycle_ms;
	u32 idx;

	struct sk_buff_head delay_q;
	struct sk_buff_head link_q;
	u32 link_bytes;			/* Mahimahi bytes in link_q */
	struct sk_buff *transit;	/* owned; head packet partly sent */
	u32 transit_left;
	struct sk_buff_head out_q;	/* sent, awaiting dequeue */
	struct sk_buff_head bypass_q;
	struct qdisc_watchdog watchdog;
	struct mmlink_stats st;
};

static const struct nla_policy mmlink_policy[TCA_MMLINK_MAX + 1] = {
	[TCA_MMLINK_BASE_NS] = { .type = NLA_U64 },
	[TCA_MMLINK_DELAY_NS] = { .type = NLA_U64 },
	[TCA_MMLINK_ORDER] = NLA_POLICY_MAX(NLA_U8, MMLINK_ORDER_LINK_DELAY),
	[TCA_MMLINK_TRACE] = NLA_POLICY_MIN_LEN(2),
	[TCA_MMLINK_LIMIT_PKTS] = { .type = NLA_U32 },
	[TCA_MMLINK_LIMIT_BYTES] = { .type = NLA_U32 },
	[TCA_MMLINK_DELAY_LIMIT] = NLA_POLICY_MIN(NLA_U32, 1),
	[TCA_MMLINK_LINK_ID] = { .type = NLA_U32 },
	[TCA_MMLINK_LOSS] = { .type = NLA_U32 },
};

static struct mmlink_cb *mmlink_cb(const struct sk_buff *skb)
{
	qdisc_cb_private_validate(skb, sizeof(struct mmlink_cb));
	return (struct mmlink_cb *)qdisc_skb_cb(skb)->data;
}

/* Times before base_ns map to ms 0; mmlink_accept() counts them as early. */
static u64 mmlink_ms(const struct mmlink_sched *q, u64 t_ns)
{
	return t_ns <= q->base_ns ? 0 : div_u64(t_ns - q->base_ns, NSEC_PER_MSEC);
}

static u64 mmlink_ms_ns(const struct mmlink_sched *q, u64 ms)
{
	return q->base_ns + ms * NSEC_PER_MSEC;
}

static u64 mmlink_next_opp(const struct mmlink_sched *q)
{
	return q->cycle_ms + q->idx;
}

/* Advance the cursor to the next millisecond that has opportunities. The
 * scan ends within one period because trace[period] != 0.
 */
static void mmlink_next_ms(struct mmlink_sched *q)
{
	do {
		if (q->idx == q->period) {
			q->idx = 0;
			q->cycle_ms += q->period;
		} else {
			q->idx++;
		}
	} while (!q->trace[q->idx]);
}

/* Mahimahi packet size, or 0 if the packet is not a parseable IPv4 packet. */
static u32 mmlink_len(const struct sk_buff *skb)
{
	struct iphdr _iph;
	const struct iphdr *iph;

	iph = skb_header_pointer(skb, skb_network_offset(skb), sizeof(_iph), &_iph);
	if (!iph || iph->version != 4)
		return 0;
	return ntohs(iph->tot_len) + MMLINK_TUN_HDR;
}

static void mmlink_flow(const struct sk_buff *skb, struct expl_rec_link *r)
{
	int off = skb_network_offset(skb);
	struct iphdr _iph;
	const struct iphdr *iph;
	struct tcphdr _th;
	const struct tcphdr *th;
	struct udphdr _uh;
	const struct udphdr *uh;

	iph = skb_header_pointer(skb, off, sizeof(_iph), &_iph);
	if (!iph || iph->version != 4)
		return;
	r->proto = iph->protocol;
	off += iph->ihl * 4;
	if (iph->protocol == IPPROTO_TCP) {
		th = skb_header_pointer(skb, off, sizeof(_th), &_th);
		if (!th)
			return;
		r->sport = ntohs(th->source);
		r->dport = ntohs(th->dest);
		r->tcp_seq = ntohl(th->seq);
		r->tcp_ack = ntohl(th->ack_seq);
		r->tcp_flags = ((const u8 *)th)[13]; /* tcp_flag_byte() */
	} else if (iph->protocol == IPPROTO_UDP) {
		uh = skb_header_pointer(skb, off, sizeof(_uh), &_uh);
		if (!uh)
			return;
		r->sport = ntohs(uh->source);
		r->dport = ntohs(uh->dest);
	}
}

static void mmlink_emit(const struct mmlink_sched *q, u16 type,
			const struct sk_buff *skb, u64 t_ns, u64 ms,
			u64 arrival_ms, u8 reason)
{
	struct expl_rec_link r = {
		.hdr = { .type = type, .len = sizeof(r), .t_ns = t_ns },
		.link_id = q->link_id,
		.bytes = mmlink_cb(skb)->len,
		.ms = ms,
		.arrival_ms = arrival_ms,
		.qlen_pkts = skb_queue_len(&q->link_q),
		.qlen_bytes = q->link_bytes,
		.reason = reason,
	};

	mmlink_flow(skb, &r);
	expl_rec_emit(&r.hdr);
}

/* Drop a packet the qdisc already accounted for. */
static void mmlink_drop(struct Qdisc *sch, struct sk_buff *skb)
{
	unsigned int len = qdisc_pkt_len(skb);

	sch->q.qlen--;
	qdisc_qstats_backlog_dec(sch, skb);
	qdisc_tree_reduce_backlog(sch, 1, len);
	qdisc_qstats_drop(sch);
	kfree_skb_reason(skb, SKB_DROP_REASON_QDISC_DROP);
}

static void mmlink_delay_enqueue(struct Qdisc *sch, struct sk_buff *skb,
				 u64 release_ns)
{
	struct mmlink_sched *q = qdisc_priv(sch);

	if (skb_queue_len(&q->delay_q) >= q->delay_limit) {
		q->st.drops_delay++;
		mmlink_emit(q, EXPL_REC_LINK_DROP, skb, release_ns,
			    mmlink_ms(q, release_ns), mmlink_ms(q, release_ns),
			    EXPL_DROP_DELAY_FULL);
		mmlink_drop(sch, skb);
		return;
	}
	mmlink_cb(skb)->t = release_ns;
	__skb_queue_tail(&q->delay_q, skb);
}

/* A packet whose last byte left on the opportunity at ms t. */
static void mmlink_depart(struct Qdisc *sch, struct sk_buff *skb, u64 t)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	u64 t_ns = mmlink_ms_ns(q, t);

	q->st.departures++;
	mmlink_emit(q, EXPL_REC_LINK_DEPART, skb, t_ns, t, mmlink_cb(skb)->t,
		    EXPL_DROP_NONE);
	if (q->order == MMLINK_ORDER_DELAY_LINK)
		__skb_queue_tail(&q->out_q, skb);
	else
		mmlink_delay_enqueue(sch, skb, t_ns + q->delay_ns);
}

/* One delivery opportunity at ms t (the inner loop of
 * LinkQueue::rationalize). Returns the bytes it carried.
 */
static u32 mmlink_deliver(struct Qdisc *sch, u64 t)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	u32 left = MMLINK_PACKET_SIZE, amount;
	struct sk_buff *skb;

	while (left) {
		if (!q->transit) {
			q->transit = __skb_dequeue(&q->link_q);
			if (!q->transit)
				break;
			q->transit_left = mmlink_cb(q->transit)->len;
			q->link_bytes -= q->transit_left;
		}
		amount = min(left, q->transit_left);
		left -= amount;
		q->transit_left -= amount;
		if (!q->transit_left) {
			skb = q->transit;
			q->transit = NULL;
			mmlink_depart(sch, skb, t);
		}
	}
	return MMLINK_PACKET_SIZE - left;
}

/* Use every opportunity at ms <= now_ms (LinkQueue::rationalize). Each
 * millisecond is processed whole, so the cursor never stops mid-millisecond.
 */
static void mmlink_rationalize(struct Qdisc *sch, u64 now_ms)
{
	struct mmlink_sched *q = qdisc_priv(sch);

	while (mmlink_next_opp(q) <= now_ms) {
		u64 t = mmlink_next_opp(q);
		u32 n = q->trace[q->idx], used = 0, i;
		struct expl_rec_opp r = {
			.hdr = { .type = EXPL_REC_LINK_OPP, .len = sizeof(r),
				 .t_ns = mmlink_ms_ns(q, t) },
			.link_id = q->link_id,
			.count = n,
			.ms = t,
		};

		for (i = 0; i < n && (q->transit || !skb_queue_empty(&q->link_q)); i++)
			used += mmlink_deliver(sch, t);
		r.used_bytes = used;
		expl_rec_emit(&r.hdr);
		mmlink_next_ms(q);
	}
}

/* A packet reaches the link stage at t_ns (LinkQueue::read_packet). */
static void mmlink_link_enqueue(struct Qdisc *sch, struct sk_buff *skb, u64 t_ns)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	u64 ms = mmlink_ms(q, t_ns);
	u32 len = mmlink_cb(skb)->len;

	mmlink_rationalize(sch, ms);
	q->st.arrivals++;
	mmlink_emit(q, EXPL_REC_LINK_ARRIVAL, skb, t_ns, ms, ms, EXPL_DROP_NONE);
	if ((q->limit_pkts && skb_queue_len(&q->link_q) + 1 > q->limit_pkts) ||
	    (q->limit_bytes && q->link_bytes + len > q->limit_bytes)) {
		q->st.drops_queue++;
		mmlink_emit(q, EXPL_REC_LINK_DROP, skb, t_ns, ms, ms,
			    EXPL_DROP_QUEUE_FULL);
		mmlink_drop(sch, skb);
		return;
	}
	mmlink_cb(skb)->t = ms;
	__skb_queue_tail(&q->link_q, skb);
	q->link_bytes += len;
}

/* Process every event up to now, in time order. Delay-line releases are
 * merged with link opportunities: a release at ms a joins the link only
 * after all opportunities at ms <= a.
 */
static void mmlink_advance(struct Qdisc *sch, u64 now)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	struct sk_buff *skb;

	if (q->order == MMLINK_ORDER_DELAY_LINK) {
		while ((skb = skb_peek(&q->delay_q)) && mmlink_cb(skb)->t <= now) {
			__skb_unlink(skb, &q->delay_q);
			mmlink_link_enqueue(sch, skb, mmlink_cb(skb)->t);
		}
	}
	mmlink_rationalize(sch, mmlink_ms(q, now));
}

/* Take ownership of one MTU-sized (or smaller) packet. */
static void mmlink_accept(struct Qdisc *sch, struct sk_buff *skb, u64 now)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	u32 len = mmlink_len(skb);

	sch->q.qlen++;
	qdisc_qstats_backlog_inc(sch, skb);
	mmlink_cb(skb)->len = len;
	if (!len || len > MMLINK_PACKET_SIZE) {
		q->st.drops_oversize++;
		mmlink_emit(q, EXPL_REC_LINK_DROP, skb, now, mmlink_ms(q, now),
			    mmlink_ms(q, now), EXPL_DROP_OVERSIZE);
		mmlink_drop(sch, skb);
		return;
	}
	/* mm-loss: an i.i.d. drop before the delay line and the link. The
	 * kernel RNG is seeded per run in Tier 1, so this is reproducible.
	 */
	if (q->loss && get_random_u32() < q->loss) {
		q->st.drops_random++;
		mmlink_emit(q, EXPL_REC_LINK_DROP, skb, now, mmlink_ms(q, now),
			    mmlink_ms(q, now), EXPL_DROP_RANDOM);
		mmlink_drop(sch, skb);
		return;
	}
	if (now < q->base_ns)
		q->st.early++;
	mmlink_advance(sch, now);
	if (q->order == MMLINK_ORDER_DELAY_LINK)
		mmlink_delay_enqueue(sch, skb, now + q->delay_ns);
	else
		mmlink_link_enqueue(sch, skb, now);
}

static int mmlink_enqueue(struct sk_buff *skb, struct Qdisc *sch,
			  struct sk_buff **to_free)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	unsigned int prev_len = qdisc_pkt_len(skb), len = 0;
	u64 now = ktime_get_ns();
	struct sk_buff *segs, *next;
	int nb = 0;

	if (skb->protocol != htons(ETH_P_IP)) {
		q->st.bypassed++;
		sch->q.qlen++;
		qdisc_qstats_backlog_inc(sch, skb);
		__skb_queue_tail(&q->bypass_q, skb);
		return NET_XMIT_SUCCESS;
	}
	if (!skb_is_gso(skb)) {
		mmlink_accept(sch, skb, now);
		return NET_XMIT_SUCCESS;
	}

	segs = skb_gso_segment(skb, netif_skb_features(skb) & ~NETIF_F_GSO_MASK);
	if (IS_ERR_OR_NULL(segs)) {
		q->st.drops_gso++;
		mmlink_cb(skb)->len = 0;
		mmlink_emit(q, EXPL_REC_LINK_DROP, skb, now, mmlink_ms(q, now),
			    mmlink_ms(q, now), EXPL_DROP_GSO);
		return qdisc_drop(skb, sch, to_free);
	}
	/* The GSO skb itself is never accounted; each segment is, on accept.
	 * Ancestors saw one packet of prev_len bytes that became nb segments.
	 */
	skb_list_walk_safe(segs, segs, next) {
		skb_mark_not_on_list(segs);
		qdisc_skb_cb(segs)->pkt_len = segs->len;
		nb++;
		len += segs->len;
		mmlink_accept(sch, segs, now);
	}
	qdisc_tree_reduce_backlog(sch, 1 - nb, prev_len - len);
	consume_skb(skb);
	return NET_XMIT_SUCCESS;
}

static u64 mmlink_next_event(const struct mmlink_sched *q)
{
	u64 next = U64_MAX;
	struct sk_buff *skb = skb_peek(&q->delay_q);

	if (skb)
		next = mmlink_cb(skb)->t;
	if (q->transit || !skb_queue_empty(&q->link_q))
		next = min(next, mmlink_ms_ns(q, mmlink_next_opp(q)));
	return next;
}

static struct sk_buff *mmlink_dequeue(struct Qdisc *sch)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	struct sk_buff *skb = __skb_dequeue(&q->bypass_q);
	u64 now, next;

	if (!skb) {
		now = ktime_get_ns();
		mmlink_advance(sch, now);
		if (q->order == MMLINK_ORDER_DELAY_LINK) {
			skb = __skb_dequeue(&q->out_q);
		} else {
			skb = skb_peek(&q->delay_q);
			if (skb && mmlink_cb(skb)->t <= now)
				__skb_unlink(skb, &q->delay_q);
			else
				skb = NULL;
		}
	}
	if (skb) {
		sch->q.qlen--;
		qdisc_qstats_backlog_dec(sch, skb);
		qdisc_bstats_update(sch, skb);
		return skb;
	}
	next = mmlink_next_event(q);
	if (next != U64_MAX)
		qdisc_watchdog_schedule_ns(&q->watchdog, next);
	return NULL;
}

static void mmlink_reset(struct Qdisc *sch)
{
	struct mmlink_sched *q = qdisc_priv(sch);

	q->st.purged += skb_queue_len(&q->delay_q) + skb_queue_len(&q->link_q) +
			skb_queue_len(&q->out_q) + !!q->transit;
	__skb_queue_purge(&q->delay_q);
	__skb_queue_purge(&q->link_q);
	__skb_queue_purge(&q->out_q);
	__skb_queue_purge(&q->bypass_q);
	kfree_skb(q->transit);
	q->transit = NULL;
	q->transit_left = 0;
	q->link_bytes = 0;
	qdisc_watchdog_cancel(&q->watchdog);
}

static int mmlink_init(struct Qdisc *sch, struct nlattr *opt,
		       struct netlink_ext_ack *extack)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_MMLINK_MAX + 1];
	u32 len;
	int err;

	/* destroy() runs even when init fails, so set up what it touches first. */
	qdisc_watchdog_init(&q->watchdog, sch);
	__skb_queue_head_init(&q->delay_q);
	__skb_queue_head_init(&q->link_q);
	__skb_queue_head_init(&q->out_q);
	__skb_queue_head_init(&q->bypass_q);

	if (!opt) {
		NL_SET_ERR_MSG_MOD(extack, "mmlink needs options");
		return -EINVAL;
	}
	err = nla_parse_nested(tb, TCA_MMLINK_MAX, opt, mmlink_policy, extack);
	if (err)
		return err;
	if (NL_REQ_ATTR_CHECK(extack, opt, tb, TCA_MMLINK_BASE_NS) ||
	    NL_REQ_ATTR_CHECK(extack, opt, tb, TCA_MMLINK_ORDER) ||
	    NL_REQ_ATTR_CHECK(extack, opt, tb, TCA_MMLINK_TRACE) ||
	    NL_REQ_ATTR_CHECK(extack, opt, tb, TCA_MMLINK_LINK_ID))
		return -EINVAL;

	q->base_ns = nla_get_u64(tb[TCA_MMLINK_BASE_NS]);
	q->delay_ns = nla_get_u64_default(tb[TCA_MMLINK_DELAY_NS], 0);
	q->order = nla_get_u8(tb[TCA_MMLINK_ORDER]);
	q->limit_pkts = nla_get_u32_default(tb[TCA_MMLINK_LIMIT_PKTS], 0);
	q->limit_bytes = nla_get_u32_default(tb[TCA_MMLINK_LIMIT_BYTES], 0);
	q->delay_limit = nla_get_u32_default(tb[TCA_MMLINK_DELAY_LIMIT],
					     MMLINK_DELAY_LIMIT_DEFAULT);
	q->link_id = nla_get_u32(tb[TCA_MMLINK_LINK_ID]);
	q->loss = nla_get_u32_default(tb[TCA_MMLINK_LOSS], 0);
	if (!q->limit_pkts && !q->limit_bytes) {
		NL_SET_ERR_MSG_MOD(extack, "droptail needs a packet or byte limit");
		return -EINVAL;
	}

	len = nla_len(tb[TCA_MMLINK_TRACE]);
	q->period = len - 1;
	q->trace = kvmemdup(nla_data(tb[TCA_MMLINK_TRACE]), len, GFP_KERNEL);
	if (!q->trace)
		return -ENOMEM;
	if (!q->trace[q->period]) {
		NL_SET_ERR_MSG_ATTR(extack, tb[TCA_MMLINK_TRACE],
				    "last trace entry must be nonzero");
		return -EINVAL;
	}
	q->cycle_ms = 0;
	q->idx = 0;
	while (!q->trace[q->idx])
		q->idx++;
	q->configured = true;
	return 0;
}

static void mmlink_destroy(struct Qdisc *sch)
{
	struct mmlink_sched *q = qdisc_priv(sch);
	struct expl_rec_link_stats r = {
		.hdr = { .type = EXPL_REC_LINK_STATS, .len = sizeof(r),
			 .t_ns = ktime_get_ns() },
		.link_id = q->link_id,
		.arrivals = q->st.arrivals,
		.departures = q->st.departures,
		.drops_queue = q->st.drops_queue,
		.drops_delay = q->st.drops_delay,
		.drops_oversize = q->st.drops_oversize,
		.drops_gso = q->st.drops_gso,
		.bypassed = q->st.bypassed,
		.early = q->st.early,
		.purged = q->st.purged,
		.drops_random = q->st.drops_random,
	};

	qdisc_watchdog_cancel(&q->watchdog);
	if (q->configured)
		expl_rec_emit(&r.hdr);
	/* reset() already purged the queues; the core calls it first. */
	kvfree(q->trace);
}

static struct Qdisc_ops mmlink_qdisc_ops __read_mostly = {
	.id = "mmlink",
	.priv_size = sizeof(struct mmlink_sched),
	.enqueue = mmlink_enqueue,
	.dequeue = mmlink_dequeue,
	.peek = qdisc_peek_dequeued,
	.init = mmlink_init,
	.reset = mmlink_reset,
	.destroy = mmlink_destroy,
	.owner = THIS_MODULE,
};
MODULE_ALIAS_NET_SCH("mmlink");

static int __init mmlink_module_init(void)
{
	return register_qdisc(&mmlink_qdisc_ops);
}

static void __exit mmlink_module_exit(void)
{
	unregister_qdisc(&mmlink_qdisc_ops);
}

module_init(mmlink_module_init);
module_exit(mmlink_module_exit);
MODULE_DESCRIPTION("Trace-driven link with Mahimahi semantics");
MODULE_LICENSE("GPL");
