#!/usr/bin/env bash
# Build the Tier 1 User-Mode Linux kernel from the vendored tree.
#
# Usage: scripts/build-uml.sh OUT_DIR [EXTRA_FRAGMENT ...]
#
# Source is third_party/linux: upstream Linux 7.0 imported unmodified in one
# commit, with our kernel patches as later commits (see third_party/README.md).
# Builds out of tree into OUT_DIR, so the source stays pristine.
#
# Produces OUT_DIR/linux (static UML binary), OUT_DIR/.config, a tree prepared
# for out-of-tree modules (make -C third_party/linux O=OUT_DIR M=...), and
# OUT_DIR/BUILD_INFO with the provenance of this build. Re-running rebuilds
# incrementally. JOBS sets make parallelism (default nproc). KERNEL_SRC
# overrides the source tree (CI's mainline tripwire); provenance then
# records the vendored tree's hash, so only use it for that tripwire.
set -euo pipefail

OUT="${1:?usage: build-uml.sh OUT_DIR [EXTRA_FRAGMENT ...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${KERNEL_SRC:-$HERE/third_party/linux}"
JOBS="${JOBS:-$(nproc)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

fragments=("$HERE/kernel/uml.config" "$@")
make -s -C "$SRC" O="$OUT" ARCH=um x86_64_defconfig
"$SRC/scripts/kconfig/merge_config.sh" -m -O "$OUT" "$OUT/.config" "${fragments[@]}" > /dev/null
make -s -C "$SRC" O="$OUT" ARCH=um olddefconfig

# merge_config only warns when Kconfig drops a requested value; fail instead.
missing=0
for frag in "${fragments[@]}"; do
    while IFS= read -r line; do
        grep -qxF -- "$line" "$OUT/.config" || { echo "not applied: $line ($frag)" >&2; missing=1; }
    done < <(grep -E '^(CONFIG_|# CONFIG_.* is not set)' "$frag")
done
# HZ has no prompt in arch/um; the value comes from our Kconfig patch.
grep -qxF "CONFIG_HZ=1000" "$OUT/.config" || { echo "CONFIG_HZ is not 1000" >&2; missing=1; }
(( missing == 0 ))

make -s -C "$SRC" O="$OUT" ARCH=um -j"$JOBS" linux modules modules_prepare

# Provenance: the git tree hash pins the exact kernel source, patches included.
dirty="$(git -C "$HERE" status --porcelain -- third_party/linux kernel | wc -l)"
{
    echo "kernel_release=$(cat "$OUT/include/config/kernel.release")"
    echo "kernel_tree=$(git -C "$HERE" rev-parse HEAD:third_party/linux)"
    echo "repo_commit=$(git -C "$HERE" rev-parse HEAD)"
    echo "source_dirty_paths=$dirty"
    echo "config_sha256=$(sha256sum "$OUT/.config" | cut -d' ' -f1)"
    echo "fragments=${fragments[*]#"$HERE/"}"
} > "$OUT/BUILD_INFO"
echo "built $OUT/linux"
cat "$OUT/BUILD_INFO"
