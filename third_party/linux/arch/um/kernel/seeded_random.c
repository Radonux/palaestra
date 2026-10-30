// SPDX-License-Identifier: GPL-2.0
/*
 * CONFIG_UML_SEEDED_RANDOM: a deterministic stand-in for host entropy, so a
 * time-travel run is reproducible from its inputs and um_seed=. This is
 * prandom (not cryptographically secure) and must only be used for
 * simulation.
 */
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/prandom.h>
#include <linux/spinlock.h>
#include <asm/archrandom.h>
#include <init.h>
#include <os.h>

static struct rnd_state um_rnd;
static bool um_rnd_seeded;
/* Protects um_rnd and um_rnd_seeded; callers may run with interrupts off. */
static DEFINE_RAW_SPINLOCK(um_rnd_lock);

static int __init um_seed_setup(char *line, int *add)
{
	u64 seed;

	if (kstrtoull(line, 0, &seed)) {
		os_warn("um_seed=%s is not a u64; using 0\n", line);
		return 0;
	}
	prandom_seed_state(&um_rnd, seed);
	um_rnd_seeded = true;
	return 0;
}

__uml_setup("um_seed=", um_seed_setup,
"um_seed=<u64>\n"
"    Seed for the deterministic randomness of CONFIG_UML_SEEDED_RANDOM\n"
"    (default 0).\n\n");

/* Context: any. Always fills all max_longs. */
size_t um_seeded_random_longs(unsigned long *v, size_t max_longs)
{
	unsigned long flags;
	size_t i;

	raw_spin_lock_irqsave(&um_rnd_lock, flags);
	if (!um_rnd_seeded) {
		prandom_seed_state(&um_rnd, 0);
		um_rnd_seeded = true;
	}
	for (i = 0; i < max_longs; i++)
		v[i] = (u64)prandom_u32_state(&um_rnd) << 32 | prandom_u32_state(&um_rnd);
	raw_spin_unlock_irqrestore(&um_rnd_lock, flags);
	return max_longs;
}
