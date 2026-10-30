#!/usr/bin/env bash
# Style-check our kernel code with the vendored checkpatch.pl --strict:
#   - the out-of-tree modules in kernel/modules, as whole files;
#   - every local patch commit on top of the Linux import, as patches.
# For patches, BAD_SIGN_OFF (our commit trailers are not kernel sign-offs)
# and PARENTHESIS_ALIGNMENT (upstream's own __uml_setup help-string style)
# are not enforced. Needs full git history.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINUX="$HERE/third_party/linux"
status=0

while read -r f; do
    (cd "$LINUX" && scripts/checkpatch.pl --strict --quiet -f "$HERE/$f") || status=1
done < <(git -C "$HERE" ls-files 'kernel/modules/*.[ch]')

import="$(git -C "$HERE" log --diff-filter=A --format=%H -- third_party/linux/Makefile | tail -1)"
[[ -n "$import" ]] || { echo "cannot find the Linux import commit (shallow clone?)" >&2; exit 1; }
while read -r commit; do
    # Strip the third_party/linux/ prefix so checkpatch sees kernel paths.
    git -C "$HERE" format-patch -1 --stdout "$commit" -- third_party/linux \
        | sed -e 's#^\([-+][-+][-+] [ab]/\)third_party/linux/#\1#' \
              -e 's#^diff --git a/third_party/linux/\(.*\) b/third_party/linux/#diff --git a/\1 b/#' \
        | (cd "$LINUX" && scripts/checkpatch.pl --strict --quiet --no-signoff \
              --ignore BAD_SIGN_OFF,PARENTHESIS_ALIGNMENT,FILE_PATH_CHANGES,COMMIT_LOG_LONG_LINE -) \
        || { echo "checkpatch failed for $commit" >&2; status=1; }
done < <(git -C "$HERE" rev-list --reverse "$import..HEAD" -- third_party/linux)
exit "$status"
