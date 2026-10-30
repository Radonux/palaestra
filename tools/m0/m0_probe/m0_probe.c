// SPDX-License-Identifier: GPL-2.0
/*
 * M0 feasibility probe: proves that an out-of-tree module loads in UML and
 * can attach to the tcp_probe tracepoint, which is not exported for direct
 * registration and must be found by name.
 *
 * Context: probe() runs wherever tcp_rcv_established() runs (softirq, or
 * process context from the socket backlog). It only bumps an atomic.
 */
#include <linux/atomic.h>
#include <linux/module.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/tracepoint.h>
#include <net/sock.h>

static struct tracepoint *tp_tcp_probe;
static atomic64_t hits = ATOMIC64_INIT(0);

static void probe(void *data, struct sock *sk, const struct sk_buff *skb)
{
	atomic64_inc(&hits);
}

static void find_tcp_probe(struct tracepoint *tp, void *priv)
{
	if (!strcmp(tp->name, "tcp_probe"))
		*(struct tracepoint **)priv = tp;
}

static int hits_get(char *buf, const struct kernel_param *kp)
{
	return sysfs_emit(buf, "%lld\n", (long long)atomic64_read(&hits));
}

static const struct kernel_param_ops hits_ops = { .get = hits_get };
module_param_cb(hits, &hits_ops, NULL, 0444);

static int __init m0_probe_init(void)
{
	for_each_kernel_tracepoint(find_tcp_probe, &tp_tcp_probe);
	if (!tp_tcp_probe)
		return -ENOENT;
	return tracepoint_probe_register(tp_tcp_probe, probe, NULL);
}

static void __exit m0_probe_exit(void)
{
	tracepoint_probe_unregister(tp_tcp_probe, probe, NULL);
	tracepoint_synchronize_unregister();
}

module_init(m0_probe_init);
module_exit(m0_probe_exit);
MODULE_DESCRIPTION("expliccit-env M0 probe: module load and tcp_probe attach");
MODULE_LICENSE("GPL");
