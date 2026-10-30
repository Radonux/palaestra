#!/usr/bin/env bash
# Build the guest side (modules, runner, root) into an existing UML build directory.
#
# Usage: scripts/build-guest.sh UML_BUILD_DIR
#
# UML_BUILD_DIR is the output of scripts/build-uml.sh. Produces, inside it:
#   modules/*.ko   the expliccit kernel modules, built against that kernel
#   rootfs/        the guest root: /init (the static runner), /modules, and
#                  empty mount points; booted read-only through hostfs
# KERNEL_SRC must match the one given to build-uml.sh, if any.
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
make -s -C "${KERNEL_SRC:-$HERE/third_party/linux}" O="$BUILD" ARCH=um M="$mods" W=1 modules

# Teacher congestion controls that the kernel config leaves out, built from
# the unmodified in-tree sources as out-of-tree modules, so the kernel image
# stays bit-identical (enabling them in kernel/uml.config changes it).
src="${KERNEL_SRC:-$HERE/third_party/linux}"
teachers="$BUILD/teachers"
rm -rf "$teachers"
mkdir -p "$teachers"
cp "$src"/net/ipv4/tcp_vegas.h "$teachers/"
for cc in cdg highspeed hybla illinois lp nv scalable vegas veno yeah; do
    cp "$src/net/ipv4/tcp_$cc.c" "$teachers/"
    echo "obj-m += tcp_$cc.o" >> "$teachers/Kbuild"
done
make -s -C "$src" O="$BUILD" ARCH=um M="$teachers" modules

target=x86_64-unknown-linux-musl
# Run cargo inside guest/ so rustup honours guest/rust-toolchain.toml (it
# resolves the toolchain from the working directory, not the manifest).
(cd "$HERE/guest" && cargo build --quiet --release --locked --target "$target")

root="$BUILD/rootfs"
rm -rf "$root"
mkdir -p "$root"/{proc,sys,dev,in,out,modules}
install -m 0755 "$HERE/guest/target/$target/release/expliccit-guest" "$root/init"
install -m 0644 "$mods"/*.ko "$teachers"/*.ko "$root/modules/"
# BBRv3, BIC, H-TCP and Westwood are modules of the kernel build itself.
install -m 0644 "$BUILD"/net/ipv4/tcp_*.ko "$root/modules/"
echo "guest ready in $root"
