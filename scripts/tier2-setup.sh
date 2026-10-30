#!/usr/bin/env bash
# One-time setup of a Tier 2 host: Ubuntu 26.04 LTS on its GA Linux 7.0
# kernel, run as root (decision D3/D4). UNRUN in this repository's CI: it
# needs a real host.
#
# Usage: sudo scripts/tier2-setup.sh OUT_DIR
#
# Holds the running kernel's packages so it cannot change under a campaign,
# builds the modules against its headers and the runner for this machine,
# and writes OUT_DIR/BUILD_INFO. Needs cargo on PATH (rustup per
# guest/rust-toolchain.toml) and, on Secure Boot hosts, signed modules.
set -euo pipefail

OUT="${1:?usage: tier2-setup.sh OUT_DIR}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release="$(uname -r)"
[[ "$EUID" -eq 0 ]] || { echo "run as root" >&2; exit 1; }
[[ "$release" == 7.0.* ]] || { echo "Tier 2 needs Ubuntu 26.04's 7.0 kernel; running $release" >&2; exit 1; }
command -v cargo > /dev/null || { echo "cargo not found (see guest/rust-toolchain.toml)" >&2; exit 1; }

apt-get install -y -q build-essential musl-tools "linux-headers-$release"
apt-mark hold "linux-image-$release" "linux-headers-$release" "linux-modules-$release"
if command -v mokutil > /dev/null && mokutil --sb-state 2> /dev/null | grep -q enabled; then
    echo "WARNING: Secure Boot is on; unsigned modules will not load" >&2
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
mods="$OUT/modules"
rm -rf "$mods"
mkdir -p "$mods"
cp "$HERE"/kernel/modules/{Kbuild,*.c,*.h} "$mods/"
make -C "/lib/modules/$release/build" M="$mods" W=1 modules

target="$(uname -m)-unknown-linux-musl"
(cd "$HERE/guest" && cargo build --quiet --release --locked --target "$target")
install -m 0755 "$HERE/guest/target/$target/release/expliccit-guest" "$OUT/expliccit-guest"

config="/boot/config-$release"
hz="$(grep -E '^CONFIG_HZ=' "$config" | cut -d= -f2)"
{
    echo "kernel_release=$release"
    echo "hz=$hz"
    echo "config_sha256=$(sha256sum "$config" | cut -d' ' -f1)"
    echo "repo_commit=$(git -C "$HERE" rev-parse HEAD)"
    echo "source_dirty_paths=$(git -C "$HERE" status --porcelain -- kernel guest | wc -l)"
} > "$OUT/BUILD_INFO"
if [[ "$hz" != 1000 ]]; then
    echo "WARNING: host HZ=$hz, Tier 1 uses 1000; patch arch/um/Kconfig to match" >&2
fi
echo "Tier 2 ready in $OUT"
cat "$OUT/BUILD_INFO"
