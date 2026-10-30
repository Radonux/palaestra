// SPDX-License-Identifier: GPL-2.0
/*
 * expl_rec: one global byte ring that the expliccit-env modules append
 * records to (expliccit_rec.h), drained by a single reader through the misc
 * device /dev/expl_rec.
 *
 * Producers call expl_rec_emit() from any context; the ring lock is taken
 * with interrupts disabled and held only for the copy. A record that does not
 * fit is dropped whole and counted in the "dropped" parameter; its sequence
 * number is still consumed, so the reader also sees the gap.
 *
 * read() never blocks: it returns what is buffered or -EAGAIN. The reader
 * polls with a sleep between drains. Under UML time-travel a blocking reader
 * woken per record would cost a guest context switch per record, and a
 * spinning reader would distort virtual time.
 * ponytail: polled reader, add a wait queue with a fill threshold if read
 * latency ever matters.
 * ponytail: one global ring per kernel; per-netns rings if Tier 2 ever runs
 * concurrent episodes on one host.
 */
#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/log2.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/spinlock.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>

#include "expl_rec.h"

static unsigned int ring_mib = 64;
module_param(ring_mib, uint, 0444);
MODULE_PARM_DESC(ring_mib, "ring size in MiB, rounded up to a power of two");

/*
 * head and tail are free-running byte counts, head - tail <= size. lock
 * protects head, tail, seq and dropped. The reader copies out of [tail, head)
 * without the lock: producers only write beyond head, and there is a single
 * reader (enforced by open), so that region is stable until tail moves.
 */
static struct {
	spinlock_t lock;	/* see above */
	u8 *buf;
	u64 size;
	u64 head;
	u64 tail;
	u32 seq;
	u64 dropped;
	atomic_t busy;
} ring = {
	.lock = __SPIN_LOCK_UNLOCKED(ring.lock),
	.busy = ATOMIC_INIT(0),
};

module_param_named(dropped, ring.dropped, ullong, 0444);
MODULE_PARM_DESC(dropped, "records dropped because the ring was full");

/* Context: any. The caller fills type, len and t_ns; seq is assigned here. */
void expl_rec_emit(struct expl_rec_hdr *rec)
{
	unsigned long flags;
	u64 len = rec->len, off, first;

	spin_lock_irqsave(&ring.lock, flags);
	rec->seq = ring.seq++;
	if (ring.size - (ring.head - ring.tail) < len) {
		ring.dropped++;
		spin_unlock_irqrestore(&ring.lock, flags);
		return;
	}
	off = ring.head & (ring.size - 1);
	first = min(len, ring.size - off);
	memcpy(ring.buf + off, rec, first);
	memcpy(ring.buf, (u8 *)rec + first, len - first);
	ring.head += len;
	spin_unlock_irqrestore(&ring.lock, flags);
}
EXPORT_SYMBOL_GPL(expl_rec_emit);

/* Context: process. */
static int expl_rec_open(struct inode *inode, struct file *file)
{
	struct expl_rec_meta meta = {
		.hdr = { .type = EXPL_REC_META, .len = sizeof(meta) },
		.magic = EXPL_REC_MAGIC,
		.abi = EXPL_REC_ABI,
		.hz = HZ,
		.ring_bytes = ring.size,
	};
	unsigned long flags;

	if (atomic_cmpxchg(&ring.busy, 0, 1))
		return -EBUSY;
	/* A new reader starts from an empty ring with its own META record. */
	spin_lock_irqsave(&ring.lock, flags);
	ring.tail = ring.head;
	spin_unlock_irqrestore(&ring.lock, flags);
	meta.hdr.t_ns = ktime_get_ns();
	expl_rec_emit(&meta.hdr);
	return stream_open(inode, file);
}

/* Context: process. */
static int expl_rec_release(struct inode *inode, struct file *file)
{
	atomic_set(&ring.busy, 0);
	return 0;
}

/* Context: process. Copies whole bytes of [tail, head); framing is the
 * reader's job. Returns -EAGAIN when the ring is empty.
 */
static ssize_t expl_rec_read(struct file *file, char __user *ubuf, size_t count,
			     loff_t *ppos)
{
	unsigned long flags;
	u64 head, tail, avail, off, first;

	spin_lock_irqsave(&ring.lock, flags);
	head = ring.head;
	tail = ring.tail;
	spin_unlock_irqrestore(&ring.lock, flags);

	avail = min_t(u64, head - tail, count);
	if (!avail)
		return -EAGAIN;
	off = tail & (ring.size - 1);
	first = min(avail, ring.size - off);
	if (copy_to_user(ubuf, ring.buf + off, first) ||
	    copy_to_user(ubuf + first, ring.buf, avail - first))
		return -EFAULT;

	spin_lock_irqsave(&ring.lock, flags);
	ring.tail += avail;
	spin_unlock_irqrestore(&ring.lock, flags);
	return avail;
}

static const struct file_operations expl_rec_fops = {
	.owner = THIS_MODULE,
	.open = expl_rec_open,
	.release = expl_rec_release,
	.read = expl_rec_read,
};

static struct miscdevice expl_rec_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "expl_rec",
	.fops = &expl_rec_fops,
	.mode = 0400,
};

static int __init expl_rec_init(void)
{
	int err;

	BUILD_BUG_ON(sizeof(struct expl_rec_meta) % 8);
	BUILD_BUG_ON(sizeof(struct expl_rec_link) != 64);
	BUILD_BUG_ON(sizeof(struct expl_rec_opp) != 40);
	BUILD_BUG_ON(sizeof(struct expl_rec_link_stats) != 96);
	BUILD_BUG_ON(sizeof(struct expl_rec_tcp_obs) != 160);

	if (!ring_mib || ring_mib > 1024)
		return -EINVAL;
	ring.size = roundup_pow_of_two((u64)ring_mib << 20);
	ring.buf = vmalloc(ring.size);
	if (!ring.buf)
		return -ENOMEM;
	err = misc_register(&expl_rec_dev);
	if (err)
		goto err_free;
	return 0;

err_free:
	vfree(ring.buf);
	return err;
}

static void __exit expl_rec_exit(void)
{
	misc_deregister(&expl_rec_dev);
	vfree(ring.buf);
}

module_init(expl_rec_init);
module_exit(expl_rec_exit);
MODULE_DESCRIPTION("expliccit-env record ring");
MODULE_LICENSE("GPL");
