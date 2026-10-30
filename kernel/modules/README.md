# expliccit-env kernel modules

Built out of tree against the matching kernel build:

    make -C third_party/linux O=$PWD/build/uml ARCH=um M=$PWD/kernel/modules W=1 modules

| Module | Role |
|---|---|
| `expl_rec` | Global record ring, read through `/dev/expl_rec`. Load first. |
| `sch_mmlink` | qdisc `mmlink`: trace-driven link plus delay with Mahimahi semantics. |
| `expl_tcpobs` | Sender observations from the `tcp:tcp_probe` tracepoint, for sockets with `SO_MARK == mark`. |
| `tcp_agent` | Congestion controls `agent` (Sage-style hold) and `agent_reno` (Orca-style, Reno base) driven by a policy through `/dev/expl_agent`. |

`expliccit_rec.h` (record layouts) and `sch_mmlink.h` (netlink attributes)
are ABIs shared with the guest runner and the host decoder. Change them only
together with `guest/` and `src/expliccit_env/records.py`, and bump
`EXPL_REC_ABI` for any record layout change.
