# Instructions for coding agents

Read `README.md` first: it is the architecture and the guarantees a change
must keep. Implementation detail is in `docs/architecture.md`.

## Git

- Push directly to `main`. Force-push is disabled; never rewrite history.
- Write commit messages in Conventional Commits form (the `git-commit` skill
  in `.claude/skills` has the rules).
- End every commit message with exactly this trailer, and no other
  `Co-Authored-By` line:

      Co-Authored-By: Joe Fang <25814618+MinecraftFuns@users.noreply.github.com>

- Never put session ids or links (for example `Claude-Session:`) in a commit
  message, and keep them out of pull request bodies too, since a squash merge
  copies the body into a commit.
- Change the kernel only as separate commits under `third_party/linux`, on
  top of the unmodified import (`git diff e8eadca -- third_party/linux`).
  Upstream kernel patches (for example Google's BBRv3 series) are imported
  with `git am`, keeping their original author and message; only the
  trailer above is added. Record each import in `third_party/README.md`.
- Do not edit `.github/workflows/sync-skills.yml` or the `.github/skills`
  submodule by hand: the file is the verbatim updater from BTreeMap/SKILLs,
  and the workflow bumps the submodule itself.

## Before pushing

Run what the change touches, and report only what actually ran:

    uv run ruff format --check src tests && uv run ruff check src tests && uv run pyright
    uv run pytest -q                                       # unit tests
    scripts/build-uml.sh build/uml && scripts/build-guest.sh build/uml
    EXPLICCIT_UML_DIR=build/uml uv run pytest -q            # adds the Tier 1 tests
    (cd guest && cargo fmt --check && for t in x86_64 aarch64; do
        cargo clippy --release --target $t-unknown-linux-musl -- -D warnings; done)
    scripts/check-kernel.sh                                # checkpatch on modules and kernel patches
    shellcheck scripts/*.sh tools/m0/*.sh
    uvx --from actionlint-py==1.7.12.25 actionlint && uvx zizmor==1.31.1 --offline .github/workflows

## Engineering rules

- Follow the `pl-theorist` and `ponytail` skills: parse untrusted input once at
  its boundary, make invalid states unrepresentable, keep the core pure, and
  write the least code that works. Mark deliberate limits with
  `ponytail: <ceiling>, <upgrade path>`.
- Never let a failed or invalid episode produce metrics or labels.
- In Tier 1 (UML, `time-travel=inf-cpu`), guest code must block or sleep,
  never spin: spinning distorts virtual time.
- Docs, comments and reports use plain English with no em dashes. Label
  anything not run as UNRUN.
