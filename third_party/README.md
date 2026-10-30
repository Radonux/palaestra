# Third-party sources

## linux/

Upstream Linux 7.0, imported unmodified in commit
`e8eadca Import Linux 7.0 source tree, unmodified` from the release tarball
<https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.0.tar.xz>
(sha256 `bb7f6d80b387c757b7d14bb93028fcb90f793c5c0d367736ee815a100b3891f0`,
matching kernel.org's `sha256sums.asc`, fetched 2026-10-30). That commit was
checked with `diff -r` against a fresh extraction of the tarball.

Every local kernel change is a later commit touching `third_party/linux`.
To list them, or the combined patch:

    git log --oneline e8eadca.. -- third_party/linux
    git diff e8eadca -- third_party/linux

### The patches we carry

| Commits | What | Why |
|---|---|---|
| `dea41ac` | `um`: HZ 1000 | arch/um hard-codes 100, which makes TCP's jiffy timers 10x coarser than on an HZ=1000 host kernel. |
| `12562c6` | `um`: `CONFIG_UML_SEEDED_RANDOM` | Bit-reproducible episodes. |
| `3cb3ca7` .. `8eb2345` (12) | BBRv3 core prerequisites | Imported unchanged with `git am` from google/bbr branch `bbr-v3-2026-09-16-01` (commits `05406b11b` .. `c552eebe7`), original authors and messages kept. Listed in `linux-imports.txt`, so our style check skips them. |
| `44162fe` | `tcp_bbr3` module | Same branch, commit `5dee74110`: BBRv3 as a separate congestion control `bbr3`, leaving BBRv1 as `bbr`. |
| `814b1d9` | 7.0 port of `tcp_bbr3` | The branch targets a newer kernel with a `cwnd_event_tx_start` op. |
| `56f58a1` | Opt-in app-limited checks | Keeps the rate samples of every non-BBRv3 congestion control as in plain 7.0. |

The branch is based on net-next of early June 2026; all 13 commits applied
to 7.0 without conflicts. With BBRv3 built as a module and not loaded,
seeded Cubic and BBRv1 episodes are byte-identical to the kernel without
the series (`docs/validation.md`).

License: GPL-2.0 (`linux/COPYING`).
