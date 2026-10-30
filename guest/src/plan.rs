//! The episode plan the host writes as `plan.json` next to the trace files.
//! `Plan::load` is the only way in: it parses and validates once, so the rest
//! of the runner works with values that cannot be out of range.
//! Mirrors `src/expliccit_env/guest_plan.py` on the host.

use crate::sys::{Context, Result, fail};
use serde::Deserialize;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

pub const PLAN_VERSION: u32 = 1;
/// Largest trace that fits one netlink attribute next to the other options.
pub const MAX_TRACE_BYTES: usize = 65_000;
/// TCP_CA_NAME_MAX in include/net/tcp.h, including the NUL.
const TCP_CA_NAME_MAX: usize = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPlan {
    plan_version: u32,
    duration_ms: u32,
    start_offset_ms: u32,
    cc: String,
    obs_mark: u32,
    ring_mib: u32,
    downlink: RawLink,
    uplink: RawLink,
    sysctls: Vec<(String, String)>,
    agent: Option<RawAgent>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgent {
    policy_file: String,
    interval_ms: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLink {
    delay_ns: u64,
    trace_file: String,
    limit_pkts: u32,
    limit_bytes: u32,
}

/// One direction of the path. Direction fixes the stage order.
pub struct Link {
    pub delay_ns: u64,
    /// Opportunities per millisecond; len >= 2, last entry nonzero.
    pub trace: Vec<u8>,
    /// Droptail limits, at least one nonzero (0 means no limit).
    pub limit_pkts: u32,
    pub limit_bytes: u32,
}

/// A sysctl written in the sender and receiver namespaces. The name is a
/// dotted path of `[a-z0-9_]` components, so it cannot leave /proc/sys.
pub struct Sysctl {
    pub path: String,
    pub value: String,
}

/// Who decides cwnd: the kernel's own congestion control, or a policy
/// driving tcp_agent ("agent" or "agent_reno") through /dev/expl_agent.
pub enum Control {
    Stock,
    Agent { policy: PathBuf, interval_ms: NonZeroU32 },
}

pub struct Plan {
    pub duration_ms: NonZeroU32,
    pub start_offset_ms: u32,
    pub cc: String,
    pub obs_mark: NonZeroU32,
    pub ring_mib: NonZeroU32,
    pub downlink: Link,
    pub uplink: Link,
    pub sysctls: Vec<Sysctl>,
    pub control: Control,
}

impl Plan {
    pub fn load(dir: &Path) -> Result<Plan> {
        let text = std::fs::read_to_string(dir.join("plan.json")).ctx("reading plan.json")?;
        let raw: RawPlan = serde_json::from_str(&text).ctx("parsing plan.json")?;
        if raw.plan_version != PLAN_VERSION {
            return fail(format!("plan_version {} != {PLAN_VERSION}", raw.plan_version));
        }
        let valid_cc = !raw.cc.is_empty()
            && raw.cc.len() < TCP_CA_NAME_MAX
            && raw.cc.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !valid_cc {
            return fail(format!("invalid congestion control name {:?}", raw.cc));
        }
        let is_agent = matches!(raw.cc.as_str(), "agent" | "agent_reno");
        let control = match (is_agent, raw.agent) {
            (false, None) => Control::Stock,
            (true, Some(a)) => Control::Agent {
                policy: dir.join(plain_file_name(&a.policy_file)?),
                interval_ms: nonzero(a.interval_ms, "agent.interval_ms")?,
            },
            (true, None) => return fail(format!("cc {} needs an agent section", raw.cc)),
            (false, Some(_)) => return fail(format!("cc {} takes no agent section", raw.cc)),
        };
        Ok(Plan {
            control,
            duration_ms: nonzero(raw.duration_ms, "duration_ms")?,
            start_offset_ms: raw.start_offset_ms,
            cc: raw.cc,
            obs_mark: nonzero(raw.obs_mark, "obs_mark")?,
            ring_mib: nonzero(raw.ring_mib, "ring_mib")?,
            downlink: Link::load(dir, raw.downlink).ctx("downlink")?,
            uplink: Link::load(dir, raw.uplink).ctx("uplink")?,
            sysctls: raw.sysctls.into_iter().map(Sysctl::parse).collect::<Result<_>>()?,
        })
    }
}

/// A file name inside the plan directory: no separators, no leading dot.
fn plain_file_name(name: &str) -> Result<&str> {
    let plain = !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        && !name.starts_with('.');
    if plain { Ok(name) } else { fail(format!("{name:?} must be a plain file name")) }
}

fn nonzero(v: u32, what: &str) -> Result<NonZeroU32> {
    NonZeroU32::new(v).map_or_else(|| fail(format!("{what} must be nonzero")), Ok)
}

impl Link {
    fn load(dir: &Path, raw: RawLink) -> Result<Link> {
        let trace = std::fs::read(dir.join(plain_file_name(&raw.trace_file)?))
            .ctx(format!("reading {}", raw.trace_file))?;
        if trace.len() < 2 || trace.len() > MAX_TRACE_BYTES {
            return fail(format!("trace has {} bytes, need 2..={MAX_TRACE_BYTES}", trace.len()));
        }
        if trace.last() == Some(&0) {
            return fail("trace must end with a nonzero entry");
        }
        if raw.limit_pkts == 0 && raw.limit_bytes == 0 {
            return fail("droptail needs a packet or byte limit");
        }
        Ok(Link { delay_ns: raw.delay_ns, trace, limit_pkts: raw.limit_pkts, limit_bytes: raw.limit_bytes })
    }
}

impl Sysctl {
    fn parse((name, value): (String, String)) -> Result<Sysctl> {
        let valid_name = name.split('.').count() >= 2
            && name.split('.').all(|c| {
                !c.is_empty() && c.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            });
        if !valid_name {
            return fail(format!("invalid sysctl name {name:?}"));
        }
        if value.is_empty() || value.bytes().any(|b| b.is_ascii_control() && b != b'\t') {
            return fail(format!("invalid value for sysctl {name}"));
        }
        Ok(Sysctl { path: format!("/proc/sys/{}", name.replace('.', "/")), value })
    }
}
