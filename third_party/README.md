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

License: GPL-2.0 (`linux/COPYING`).
