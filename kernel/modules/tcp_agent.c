// SPDX-License-Identifier: GPL-2.0
/*
 * tcp_agent: congestion controls actuated by a userspace policy.
 *
 * A policy (Orca- or Sage-style agent, or an exported expliCCit policy)
 * observes the flow itself (TCP_INFO) and writes struct expl_agent_action
 * to /dev/expl_agent. Two controllers apply it:
 *
 *   "agent"       Sage-style. cong_control holds snd_cwnd and the pacing rate
 *                 at the latest action on every delivery. Until the first
 *                 action the initial window is held.
 *   "agent_reno"  Orca-style. Each new action sets snd_cwnd once; between
 *                 actions Reno grows the window and the core stack handles
 *                 loss. Orca proper runs Cubic underneath; Cubic's internals
 *                 are not exported, so this base is Reno.
 *                 ponytail: Reno base, vendor Cubic's update if Orca
 *                 fidelity needs it.
 *
 * Every accepted action is reported to expl_rec as AGENT_ACTION.
 * ponytail: one global action for all agent sockets (one agent flow per
 * episode); key actions by socket if episodes ever carry several.
 *
 * Locking: action_lock (irqsave) guards the published action. Readers run in
 * softirq or in socket-backlog processing (BH enabled); the writer runs in
 * process context. Socket state is only touched from the CC hooks, which run
 * with the socket locked.
 */
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/spinlock.h>
#include <linux/uaccess.h>
#include <net/tcp.h>

#include "expl_rec.h"

static DEFINE_SPINLOCK(action_lock);	/* guards action and action_seq */
static struct expl_agent_action action;
static u32 action_seq;			/* 0: no action yet */

struct agent_priv {
	u32 applied_seq;		/* agent_reno: last action applied */
};

static u32 agent_current(struct expl_agent_action *out)
{
	unsigned long flags;
	u32 seq;

	spin_lock_irqsave(&action_lock, flags);
	*out = action;
	seq = action_seq;
	spin_unlock_irqrestore(&action_lock, flags);
	return seq;
}

static void agent_set_pacing(struct sock *sk, u64 rate)
{
	u64 max = READ_ONCE(sk->sk_max_pacing_rate);

	if (!rate) {
		WRITE_ONCE(sk->sk_pacing_rate, max);
		return;
	}
	/* Ask the stack to pace internally, as BBR does (unless fq paces). */
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
	WRITE_ONCE(sk->sk_pacing_rate, min(rate, max));
}

static void agent_apply(struct sock *sk, const struct expl_agent_action *a)
{
	struct tcp_sock *tp = tcp_sk(sk);

	tcp_snd_cwnd_set(tp, min(a->cwnd, tp->snd_cwnd_clamp));
	agent_set_pacing(sk, a->pacing_rate);
}

static void agent_init(struct sock *sk)
{
	struct agent_priv *p = inet_csk_ca(sk);

	BUILD_BUG_ON(sizeof(struct agent_priv) > ICSK_CA_PRIV_SIZE);
	p->applied_seq = 0;
}

/* "agent": hold the latest action. */
static void agent_cong_control(struct sock *sk, u32 ack, int flag,
			       const struct rate_sample *rs)
{
	struct expl_agent_action a;

	if (agent_current(&a))
		agent_apply(sk, &a);
}

/* "agent_reno": apply each new action once, at the next ACK of any kind
 * (cong_avoid alone would miss actions while not cwnd-limited); Reno runs
 * as cong_avoid in between.
 */
static void agent_reno_in_ack_event(struct sock *sk, u32 flags)
{
	struct agent_priv *p = inet_csk_ca(sk);
	struct expl_agent_action a;
	u32 seq = agent_current(&a);

	if (seq != p->applied_seq) {
		p->applied_seq = seq;
		agent_apply(sk, &a);
	}
}

static u32 agent_ssthresh(struct sock *sk)
{
	return max(tcp_snd_cwnd(tcp_sk(sk)) >> 1U, 2U);
}

static u32 agent_undo_cwnd(struct sock *sk)
{
	return tcp_snd_cwnd(tcp_sk(sk));
}

static struct tcp_congestion_ops tcp_agent __read_mostly = {
	.init = agent_init,
	.ssthresh = agent_ssthresh,
	.cong_control = agent_cong_control,
	.undo_cwnd = agent_undo_cwnd,
	.owner = THIS_MODULE,
	.name = "agent",
};

static struct tcp_congestion_ops tcp_agent_reno __read_mostly = {
	.init = agent_init,
	.ssthresh = tcp_reno_ssthresh,
	.in_ack_event = agent_reno_in_ack_event,
	.cong_avoid = tcp_reno_cong_avoid,
	.undo_cwnd = tcp_reno_undo_cwnd,
	.owner = THIS_MODULE,
	.name = "agent_reno",
};

/* Context: process. Untrusted input: one whole, valid action per write. */
static ssize_t agent_write(struct file *file, const char __user *buf,
			   size_t count, loff_t *ppos)
{
	struct expl_rec_agent_action r = {
		.hdr = { .type = EXPL_REC_AGENT_ACTION, .len = sizeof(r) },
	};
	struct expl_agent_action a;
	unsigned long flags;

	if (count != sizeof(a))
		return -EINVAL;
	if (copy_from_user(&a, buf, sizeof(a)))
		return -EFAULT;
	if (a.abi != EXPL_AGENT_ABI || a.cwnd < 2 || a.cwnd > EXPL_AGENT_MAX_CWND ||
	    a.pacing_rate > EXPL_AGENT_MAX_PACING)
		return -EINVAL;

	spin_lock_irqsave(&action_lock, flags);
	action = a;
	r.action_seq = ++action_seq;
	spin_unlock_irqrestore(&action_lock, flags);

	r.hdr.t_ns = ktime_get_ns();
	r.cwnd = a.cwnd;
	r.pacing_rate = a.pacing_rate;
	expl_rec_emit(&r.hdr);
	return count;
}

static const struct file_operations agent_fops = {
	.owner = THIS_MODULE,
	.write = agent_write,
};

static struct miscdevice agent_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "expl_agent",
	.fops = &agent_fops,
	.mode = 0200,
};

static int __init tcp_agent_init(void)
{
	int err;

	BUILD_BUG_ON(sizeof(struct expl_rec_agent_action) != 32);
	err = tcp_register_congestion_control(&tcp_agent);
	if (err)
		return err;
	err = tcp_register_congestion_control(&tcp_agent_reno);
	if (err)
		goto err_agent;
	err = misc_register(&agent_dev);
	if (err)
		goto err_reno;
	return 0;

err_reno:
	tcp_unregister_congestion_control(&tcp_agent_reno);
err_agent:
	tcp_unregister_congestion_control(&tcp_agent);
	return err;
}

static void __exit tcp_agent_exit(void)
{
	misc_deregister(&agent_dev);
	tcp_unregister_congestion_control(&tcp_agent_reno);
	tcp_unregister_congestion_control(&tcp_agent);
}

module_init(tcp_agent_init);
module_exit(tcp_agent_exit);
MODULE_DESCRIPTION("Congestion control actuated by a userspace policy");
MODULE_LICENSE("GPL");
