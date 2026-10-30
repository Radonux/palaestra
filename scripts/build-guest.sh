#!/usr/bin/env bash
# Build the guest side of Tier 1 into an existing UML build directory.
#
# Usage: scripts/build-guest.sh UML_BUILD_DIR
#
# UML_BUILD_DIR is the output of scripts/build-uml.sh. Produces, inside it:
#   modules/*.ko   the expliccit kernel modules, built against that kernel
#   rootfs/        the guest root: /init (the static runner), /modules, and
#                  empty mount points; booted read-only through hostfs
set -euo pipefail

BUILD="${1:?usage: build-guest.sh UML_BUILD_DIR}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$(cd "$BUILD" && pwd)"
[[ -x "$BUILD/linux" ]] || { echo "$BUILD/linux missing; run build-uml.sh first" >&2; exit 1; }

# Out-of-tree modules write objects next to their sources; build a copy so
# the repository stays clean.
mods="$BUILD/modules"
rm -rf "$mods"
mkdir -p "$mods"
cp "$HERE"/kernel/modules/{Kbuild,*.c,*.h} "$mods/"
make -s -C "$HERE/third_party/linux" O="$BUILD" ARCH=um M="$mods" W=1 modules

target=x86_64-unknown-linux-musl
cargo build --quiet --release --locked --manifest-path "$HERE/guest/Cargo.toml" --target "$target"

root="$BUILD/rootfs"
rm -rf "$root"
mkdir -p "$root"/{proc,sys,dev,in,out,modules}
install -m 0755 "$HERE/guest/target/$target/release/expliccit-guest" "$root/init"
install -m 0644 "$mods"/*.ko "$root/modules/"
echo "guest ready in $root"
