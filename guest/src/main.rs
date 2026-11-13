//! expliccit-guest: runs one episode and writes its records and result.
//!
//! As PID 1 of a UML guest (`init=/init`), it mounts the pseudo filesystems,
//! mounts the host directories named by the kernel command-line variables
//! `expl_in` (plan) and `expl_out` (results) on /in and /out, loads the
//! modules from /modules, runs the episode, writes result.json and powers
//! off.
//!
//! Topology (one veth pair per hop, all in fresh namespaces):
//!
//!     snd 10.0.0.1 s0 -- p0 10.0.0.2 [path] 10.0.1.2 p1 -- r0 10.0.1.1 rcv
//!
//! with the downlink mmlink (delay, then link) as root qdisc of p1 and the
//! uplink mmlink (link, then delay) as root qdisc of p0. Crossing into the
//! path namespace orphans each skb, so the sender's TCP Small Queues see the
//! bottleneck as remote, like a Mahimahi shell.

mod cross;
mod netlink;
mod netns;
mod plan;
mod recorder;
mod step;
mod sys;
mod traffic;
mod truth;

use netlink::{Mmlink, Order, Socket};
use netns::Netns;
use plan::{Control, CrossFlow, Link, Plan};
use serde::Serialize;
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::process::ExitCode;
use sys::{Context, Result, fail};

const RESULT_VERSION: u32 = 7;
const SND_ADDR: [u8; 4] = [10, 0, 0, 1];
const PATH_SND_ADDR: [u8; 4] = [10, 0, 0, 2];
const RCV_ADDR: [u8; 4] = [10, 0, 1, 1];
const PATH_RCV_ADDR: [u8; 4] = [10, 0, 1, 2];
const LINK_ID_DOWN: u32 = 1;
const LINK_ID_UP: u32 = 2;
/// TCP settings of the setups this environment reproduces, Sage's
/// set_sysctl.sh and ccBench's system_setup.sh (the same lines in both),
/// written in the sender's and receiver's namespaces before the plan's own
/// sysctls (which may override them). Linux sizes the buffer limits from
/// memory, which in a 1 GiB guest caps the send buffer at 4 MB and so the
/// data a flow can keep in flight. tcp_low_latency, also in both scripts,
/// has had no effect since Linux 4.14.
const TCP_SYSCTLS: [(&str, &str); 4] = [
    ("/proc/sys/net/ipv4/tcp_rmem", "4096 87380 67108864"),
    ("/proc/sys/net/ipv4/tcp_wmem", "4096 65536 67108864"),
    ("/proc/sys/net/ipv4/tcp_no_metrics_save", "1"),
    ("/proc/sys/net/ipv4/tcp_autocorking", "0"),
];

/// Raised so the veth receive backlog never drops a burst released by the
/// qdisc; softnet drops are still checked and reported.
const NETDEV_MAX_BACKLOG: &str = "100000";

#[derive(Serialize)]
struct Outcome {
    result_version: u32,
    status: &'static str,
    error: Option<String>,
    kernel_release: String,
    kernel_version: String,
    episode: Option<Report>,
}

#[derive(Serialize)]
struct Report {
    base_ns: u64,
    end_ns: u64,
    duration_ms: u32,
    cc_requested: String,
    cc_verified: String,
    bytes_sent: u64,
    bytes_received: u64,
    records: BTreeMap<&'static str, u64>,
    record_bytes: u64,
    ring_dropped: u64,
    softnet_dropped: u64,
    netdev_max_backlog: &'static str,
    agent_actions: u64,
    /// Per cross flow, in plan order: bytes its sender wrote, the congestion
    /// control its socket ran ("" for UDP), when its sender accepted the
    /// connection (CLOCK_MONOTONIC ns as base_ns; null for UDP, or a TCP
    /// flow whose handshake did not complete before its stop), and the
    /// steps written to steps-cross<i>.bin and actions taken (0 unless a
    /// policy drove it).
    cross_bytes_sent: Vec<u64>,
    cross_cc: Vec<String>,
    cross_connected_ns: Vec<Option<u64>>,
    cross_steps: Vec<u64>,
    cross_actions: Vec<u64>,
    /// Steps written to steps.bin (0 when the step loop did not run).
    steps: u64,
}

fn main() -> ExitCode {
    // SAFETY: getpid has no preconditions.
    if unsafe { libc::getpid() } != 1 {
        eprintln!("expliccit-guest runs only as PID 1 of a UML guest (init=/init)");
        return ExitCode::from(2);
    }
    let out = Path::new("/out");
    write_outcome(out, boot().and_then(|()| episode(Path::new("/in"), out, Path::new("/modules"))));
    sys::power_off()
}

fn write_outcome(out: &Path, result: Result<Report>) {
    let (kernel_release, kernel_version) = sys::uname();
    let (status, error, episode) = match result {
        Ok(r) => ("ok", None, Some(r)),
        Err(e) => ("failed", Some(e.0), None),
    };
    let outcome =
        Outcome { result_version: RESULT_VERSION, status, error, kernel_release, kernel_version, episode };
    let text = serde_json::to_string_pretty(&outcome)
        .unwrap_or_else(|e| format!("{{\"status\":\"failed\",\"error\":\"{e}\"}}"));
    if let Err(e) = std::fs::write(out.join("result.json"), text) {
        eprintln!("expliccit-guest: cannot write result.json: {e}");
    }
}

/// PID 1 duties in a UML guest.
fn boot() -> Result<()> {
    sys::mount("proc", "/proc", "proc", None).ctx("mounting /proc")?;
    sys::mount("sysfs", "/sys", "sysfs", None).ctx("mounting /sys")?;
    sys::mount("devtmpfs", "/dev", "devtmpfs", None).ctx("mounting /dev")?;
    for (var, dir) in [("expl_in", "/in"), ("expl_out", "/out")] {
        let host = std::env::var(var).ctx(format!("kernel command line lacks {var}"))?;
        sys::mount("none", dir, "hostfs", Some(&host)).ctx(format!("mounting {host} on {dir}"))?;
    }
    Ok(())
}

fn load_modules(dir: &Path, plan: &Plan) -> Result<()> {
    let mut mods: Vec<(&str, String)> = vec![
        ("expl_rec.ko", format!("ring_mib={}", plan.ring_mib)),
        ("sch_mmlink.ko", String::new()),
        ("expl_tcpobs.ko", format!("mark={}", plan.obs_mark)),
        ("tcp_agent.ko", String::new()),
    ];
    // Loaded only when a flow has its own RTT, as the controllers below.
    if !extra_delays(plan)?.is_empty() {
        mods.push(("sch_flowdelay.ko", String::new()));
    }
    // A congestion control built as a module (BBRv3 and the extra teachers)
    // is loaded only when an episode uses it, so every other episode runs
    // without its code in the kernel (docs/validation.md). Veno and YeAH
    // use Vegas's exported functions, so Vegas comes first.
    let cross = plan.cross.iter().filter_map(|c| match c {
        CrossFlow::Tcp { cc, .. } => Some(cc.as_str()),
        CrossFlow::Udp { .. } => None,
    });
    let mut wanted: Vec<String> = Vec::new();
    for cc in std::iter::once(plan.cc.as_str()).chain(cross) {
        let deps: &[&str] = if matches!(cc, "veno" | "yeah") { &["vegas"] } else { &[] };
        for name in deps.iter().copied().chain(std::iter::once(cc)) {
            let file = format!("tcp_{name}.ko");
            let loaded = wanted.contains(&file) || mods.iter().any(|(m, _)| *m == file);
            if dir.join(&file).exists() && !loaded {
                wanted.push(file);
            }
        }
    }
    for file in &wanted {
        mods.push((file.as_str(), String::new()));
    }
    for (name, params) in mods {
        sys::finit_module(&dir.join(name), &params).ctx(format!("loading {name}"))?;
    }
    Ok(())
}

/// (sender port, extra delay) of each TCP cross flow with its own RTT. The
/// receiver's egress delays those flows' ACKs (kernel/modules/sch_flowdelay.c).
fn extra_delays(plan: &Plan) -> Result<Vec<(u16, u64)>> {
    let mut out = Vec::new();
    for (i, flow) in plan.cross.iter().enumerate() {
        if let CrossFlow::Tcp { extra_delay_ns, .. } = flow {
            if *extra_delay_ns > 0 {
                out.push((cross::port(i)?, *extra_delay_ns));
            }
        }
    }
    Ok(out)
}

/// tcp_mem (low, pressure, high, in 4 KiB pages) at 1/4, 3/8 and 1/2 of
/// the guest's memory.
fn tcp_mem() -> Result<String> {
    let info = std::fs::read_to_string("/proc/meminfo").ctx("reading /proc/meminfo")?;
    let Some(kib) = info
        .lines()
        .find_map(|l| l.strip_prefix("MemTotal:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
    else {
        return fail("no MemTotal in /proc/meminfo");
    };
    let pages = kib / 4;
    Ok(format!("{} {} {}", pages / 4, pages * 3 / 8, pages / 2))
}

fn ifindex(name: &str) -> Result<u32> {
    let c = CString::new(name).ctx("interface name")?;
    // SAFETY: c is NUL-terminated and alive for the call.
    match unsafe { libc::if_nametoindex(c.as_ptr()) } {
        0 => fail(format!("no interface {name}")),
        i => Ok(i),
    }
}

/// Bring up lo and the given addresses, set the default route and sysctls.
fn configure(
    ns: &Netns,
    home: &Netns,
    addrs: &[(&str, [u8; 4])],
    gateway: Option<[u8; 4]>,
    sysctls: &[(String, String)],
) -> Result<()> {
    ns.within(home, || {
        let mut nl = Socket::open()?;
        nl.link_up("lo")?;
        for &(dev, addr) in addrs {
            nl.add_addr(ifindex(dev)?, addr, 24)?;
            nl.link_up(dev)?;
        }
        if let Some(gw) = gateway {
            nl.add_default_route(gw)?;
        }
        for (path, value) in sysctls {
            std::fs::write(path, value).ctx(format!("writing {path}"))?;
        }
        Ok(())
    })
}

fn mmlink<'a>(link: &'a Link, link_id: u32, order: Order, base_ns: u64) -> Mmlink<'a> {
    Mmlink {
        link_id,
        order,
        base_ns,
        delay_ns: link.delay_ns,
        trace_path: &link.trace_path,
        limit_pkts: link.limit_pkts,
        limit_bytes: link.limit_bytes,
        loss: link.loss,
    }
}

fn read_u64(path: &str) -> Result<u64> {
    let s = std::fs::read_to_string(path).ctx(format!("reading {path}"))?;
    s.trim().parse().ctx(format!("parsing {path}"))
}

/// Sum of the "dropped" column (second, hex) of /proc/net/softnet_stat.
fn softnet_dropped() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/net/softnet_stat").ctx("reading softnet_stat")?;
    text.lines()
        .map(|l| {
            let col = l.split_whitespace().nth(1).unwrap_or("0");
            u64::from_str_radix(col, 16).ctx("parsing softnet_stat")
        })
        .sum()
}

fn episode(plan_dir: &Path, out: &Path, module_dir: &Path) -> Result<Report> {
    let plan = Plan::load(plan_dir)?;
    load_modules(module_dir, &plan)?;
    std::fs::write("/proc/sys/net/core/netdev_max_backlog", NETDEV_MAX_BACKLOG).ctx("netdev_max_backlog")?;
    // All sockets of the guest, sender and receiver alike, share one TCP
    // memory budget, which Linux sets to 4.7 / 6.25 / 9.4% of memory (44,
    // 59 and 88 MB here); a host many times larger never reaches it. Here
    // it starts to press at 3/8 of the guest's memory.
    std::fs::write("/proc/sys/net/ipv4/tcp_mem", tcp_mem()?).ctx("tcp_mem")?;
    // The same scripts' socket buffer maxima, global and read-only in other
    // namespaces; they bound SO_SNDBUF and SO_RCVBUF, which nothing here sets.
    for name in ["rmem_max", "wmem_max"] {
        std::fs::write(format!("/proc/sys/net/core/{name}"), "134217728").ctx(name)?;
    }
    // The step loops' true state comes from the links' records, so it
    // exists whenever steps run, for the observed flow or a cross agent.
    let n_cross_agents = plan.cross.iter().filter(|c| c.agent().is_some()).count();
    let cross_agents = n_cross_agents > 0;
    // Several step loops: each lets its instant settle before the truth.
    let settle = n_cross_agents + usize::from(plan.steps) > 1;
    let delays = extra_delays(&plan)?;
    let link_truth = (plan.steps || cross_agents).then(|| {
        let d = &plan.downlink;
        truth::LinkTruth::new(truth::Path {
            end_ms: u64::from(plan.duration_ms.get()),
            trace: d.trace.clone(),
            buffer_pkts: d.limit_pkts,
            buffer_bytes: d.limit_bytes,
            one_way_ns: d.delay_ns,
            loss: d.loss,
            extra_rtt_ns: delays.clone(),
        })
    });
    let (recorder, truth) = recorder::Recorder::start(out, link_truth)?;

    let home = Netns::current()?;
    let snd = Netns::create(&home)?;
    let path = Netns::create(&home)?;
    let rcv = Netns::create(&home)?;
    // Without IPv6 nothing but our traffic crosses the links. Devices take
    // the namespace's defaults when created, so this precedes the veths.
    let no_v6 = "/proc/sys/net/ipv6/conf/default/disable_ipv6";
    for ns in [&snd, &path, &rcv] {
        ns.within(&home, || {
            if Path::new(no_v6).exists() { std::fs::write(no_v6, "1").ctx(no_v6) } else { Ok(()) }
        })?;
    }
    let mut nl = Socket::open()?;
    nl.veth("s0", snd.fd(), "p0", path.fd())?;
    nl.veth("r0", rcv.fd(), "p1", path.fd())?;
    drop(nl);

    let sysctls: Vec<(String, String)> = TCP_SYSCTLS
        .iter()
        .map(|&(p, v)| (p.to_owned(), v.to_owned()))
        .chain(plan.sysctls.iter().map(|s| (s.path.clone(), s.value.clone())))
        .collect();
    configure(&snd, &home, &[("s0", SND_ADDR)], Some(PATH_SND_ADDR), &sysctls)?;
    configure(&rcv, &home, &[("r0", RCV_ADDR)], Some(PATH_RCV_ADDR), &sysctls)?;
    let forward = [("/proc/sys/net/ipv4/ip_forward".to_owned(), "1".to_owned())];
    configure(&path, &home, &[("p0", PATH_SND_ADDR), ("p1", PATH_RCV_ADDR)], None, &forward)?;

    if !delays.is_empty() {
        rcv.within(&home, || Socket::open()?.add_flowdelay(ifindex("r0")?, &delays))?;
    }

    // Trace ms 0 starts on a whole millisecond after the setup margin.
    let now = sys::monotonic_ns();
    let base_ns = now.div_ceil(1_000_000) * 1_000_000 + u64::from(plan.start_offset_ms) * 1_000_000;
    let (p0, p1) = path.within(&home, || {
        let (p0, p1) = (ifindex("p0")?, ifindex("p1")?);
        let mut nl = Socket::open()?;
        nl.add_mmlink(p1, &mmlink(&plan.downlink, LINK_ID_DOWN, Order::DelayLink, base_ns))?;
        nl.add_mmlink(p0, &mmlink(&plan.uplink, LINK_ID_UP, Order::LinkDelay, base_ns))?;
        Ok((p0, p1))
    })?;

    let end_ns = base_ns + u64::from(plan.duration_ms.get()) * 1_000_000;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Each agent cross flow's step loop, set up (policy loaded, files open)
    // before any traffic starts; it runs over the flow's own window.
    let at = |ms: u32| base_ns + u64::from(ms) * 1_000_000;
    let mut loops = Vec::with_capacity(plan.cross.len());
    for (i, flow) in plan.cross.iter().enumerate() {
        loops.push(match (flow, flow.agent(), &truth) {
            (CrossFlow::Tcp { start_ms, stop_ms, .. }, Some(a), Some(truth)) => Some(step::Loop::new(
                &out.join(format!("steps-cross{i}.bin")),
                Some(
                    step::Policy::load(&a.policy, a.noise, a.mode, a.period_ms)
                        .ctx(format!("cross flow {i}"))?,
                ),
                truth.clone(),
                cross::port(i)?,
                at(*start_ms),
                base_ns,
                at(*stop_ms),
                std::sync::Arc::clone(&stop),
                settle,
            )?),
            _ => None,
        });
    }
    let sender = snd.within(&home, || traffic::start_sender(&plan.cc, plan.obs_mark.get()))?;
    let cross = cross::start(&plan.cross, loops, base_ns, &snd, &rcv, &home)?;
    sys::sleep_until(base_ns);
    let receiver = rcv.within(&home, traffic::start_receiver)?;
    let Ok(conn) = sender.conn.recv() else {
        return traffic::join(sender.handle, "sender").and_then(|_| fail("sender exited before accepting"));
    };
    let steps = match (&truth, plan.steps) {
        (Some(truth), true) => {
            let policy = match &plan.control {
                Control::Stock => None,
                Control::Agent(a) => Some(step::Policy::load(&a.policy, a.noise, a.mode, a.period_ms)?),
            };
            let lp = step::Loop::new(
                &out.join("steps.bin"),
                policy,
                truth.clone(),
                traffic::SENDER_ADDR.port(),
                base_ns,
                base_ns,
                end_ns,
                std::sync::Arc::clone(&stop),
                settle,
            )?;
            Some(step::spawn("steps", lp, std::sync::Arc::clone(&conn))?)
        }
        _ => None,
    };
    sys::sleep_until(end_ns);
    stop.store(true, std::sync::atomic::Ordering::Release);
    let (steps, agent_actions) = match steps {
        Some(h) => traffic::join(h, "steps").map(|s| (s.count, s.actions))?,
        None => (0, 0),
    };
    sys::shutdown(std::os::fd::AsFd::as_fd(&*conn));
    sys::shutdown(std::os::fd::AsFd::as_fd(&*receiver.conn));
    let sent = traffic::join(sender.handle, "sender")?;
    let received = traffic::join(receiver.handle, "receiver")?;
    let flows = cross.finish()?;

    // Deleting the qdiscs emits their LINK_STATS records.
    path.within(&home, || {
        let mut nl = Socket::open()?;
        nl.del_root_qdisc(p1)?;
        nl.del_root_qdisc(p0)
    })?;
    let ring_dropped = read_u64("/sys/module/expl_rec/parameters/dropped")?;
    let softnet_dropped = softnet_dropped()?;
    let summary = recorder.finish()?;
    if sent.cc != plan.cc {
        return fail(format!("socket runs {:?}, plan asked for {:?}", sent.cc, plan.cc));
    }
    Ok(Report {
        base_ns,
        end_ns,
        duration_ms: plan.duration_ms.get(),
        cc_requested: plan.cc.clone(),
        cc_verified: sent.cc,
        bytes_sent: sent.bytes,
        bytes_received: received,
        records: summary.counts,
        record_bytes: summary.bytes,
        ring_dropped,
        softnet_dropped,
        netdev_max_backlog: NETDEV_MAX_BACKLOG,
        agent_actions,
        cross_bytes_sent: flows.iter().map(|f| f.sent).collect(),
        cross_cc: flows.iter().map(|f| f.cc.clone()).collect(),
        cross_connected_ns: flows.iter().map(|f| f.connected_ns).collect(),
        cross_steps: flows.iter().map(|f| f.steps).collect(),
        cross_actions: flows.iter().map(|f| f.actions).collect(),
        steps,
    })
}
