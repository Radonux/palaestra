//! Closed-loop policy for the tcp_agent controllers. Every interval of
//! virtual time: read TCP_INFO from the sender socket, build the tcpinfo-v1
//! features, run the ONNX policy, write the action to /dev/expl_agent.
//! The loop sleeps on an absolute schedule, so inference costs no virtual
//! time under time-travel=inf-cpu and nothing ever spins.
//!
//! Policy contract (mirrors src/expliccit_env/policy.py):
//!   input  float32 [1, 8], tcpinfo-v1 in this order:
//!          srtt_ms, min_rtt_ms, delivery_rate_mbps, cwnd_pkts, unacked_pkts,
//!          lost_pkts, retrans_in_interval, acked_mbps_in_interval
//!   output float32 [1, 2]: cwnd_pkts, pacing_mbps (0 = unpaced)
//! The cwnd is rounded and must lie in [2, 2^20]; pacing must be in
//! [0, 2^40 bytes/s]. Anything else (including NaN) fails the episode.

use crate::sys::{self, Context, Result, fail};
use std::fs::File;
use std::io::Write;
use std::net::TcpStream;
use std::os::fd::AsFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use tract_onnx::prelude::*;

pub const FEATURES: usize = 8;
const AGENT_ABI: u32 = 1;
const MAX_CWND: f32 = 1_048_576.0;
const MAX_PACING_BYTES: f64 = 1_099_511_627_776.0;

type Plan = TypedRunnableModel<TypedModel>;

pub struct Policy {
    plan: Plan,
}

impl Policy {
    pub fn load(path: &Path) -> Result<Policy> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .and_then(|m| m.with_input_fact(0, f32::fact([1, FEATURES]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .ctx(format!("loading ONNX policy {}", path.display()))?;
        Ok(Policy { plan })
    }

    fn act(&self, x: [f32; FEATURES]) -> Result<(f32, f32)> {
        let input: Tensor =
            tract_ndarray::Array2::from_shape_vec((1, FEATURES), x.to_vec()).ctx("policy input")?.into();
        let out = self.plan.run(tvec!(input.into())).ctx("running policy")?;
        let Some(first) = out.first() else { return fail("policy returned no output") };
        let view = first.to_array_view::<f32>().ctx("policy output type")?;
        let v: Vec<f32> = view.iter().copied().collect();
        match v.as_slice() {
            [cwnd, pacing] => Ok((*cwnd, *pacing)),
            _ => fail(format!("policy output has {} values, expected 2", v.len())),
        }
    }
}

/// The TCP_INFO fields tcpinfo-v1 uses (offsets from include/uapi/linux/tcp.h).
struct TcpInfo {
    unacked: u32,
    lost: u32,
    rtt_us: u32,
    snd_cwnd: u32,
    total_retrans: u32,
    bytes_acked: u64,
    min_rtt_us: u32,
    delivery_rate: u64,
}

fn tcp_info(stream: &TcpStream) -> Result<TcpInfo> {
    let mut b = [0u8; 248];
    let n =
        sys::getsockopt_bytes(stream.as_fd(), libc::IPPROTO_TCP, libc::TCP_INFO, &mut b).ctx("TCP_INFO")?;
    if n < 168 {
        return fail(format!("TCP_INFO returned {n} bytes"));
    }
    let u32_at = |o: usize| b.get(o..o + 4).and_then(|s| s.try_into().ok()).map_or(0, u32::from_ne_bytes);
    let u64_at = |o: usize| b.get(o..o + 8).and_then(|s| s.try_into().ok()).map_or(0, u64::from_ne_bytes);
    Ok(TcpInfo {
        unacked: u32_at(24),
        lost: u32_at(32),
        rtt_us: u32_at(68),
        snd_cwnd: u32_at(80),
        total_retrans: u32_at(100),
        bytes_acked: u64_at(120),
        min_rtt_us: u32_at(148),
        delivery_rate: u64_at(160),
    })
}

fn action_bytes(cwnd: f32, pacing_mbps: f32) -> Result<[u8; 16]> {
    let cwnd = cwnd.round();
    if !(2.0..=MAX_CWND).contains(&cwnd) {
        return fail(format!("policy cwnd {cwnd} outside [2, {MAX_CWND}]"));
    }
    let pacing = f64::from(pacing_mbps) * 1e6 / 8.0;
    if !(0.0..=MAX_PACING_BYTES).contains(&pacing) {
        return fail(format!("policy pacing {pacing_mbps} Mbit/s out of range"));
    }
    let mut b = [0u8; 16];
    b[0..4].copy_from_slice(&AGENT_ABI.to_ne_bytes());
    // Range-checked above, so the casts are exact.
    b[4..8].copy_from_slice(&(cwnd as u32).to_ne_bytes());
    b[8..16].copy_from_slice(&(pacing as u64).to_ne_bytes());
    Ok(b)
}

/// Act at start + k * interval for k >= 1 while that time is before end.
pub fn spawn(
    stream: Arc<TcpStream>,
    policy: Policy,
    start_ns: u64,
    interval_ns: u64,
    end_ns: u64,
    stop: Arc<AtomicBool>,
) -> Result<JoinHandle<Result<u64>>> {
    let mut dev = File::options().write(true).open("/dev/expl_agent").ctx("opening /dev/expl_agent")?;
    std::thread::Builder::new()
        .name("agent".into())
        .spawn(move || -> Result<u64> {
            let mut actions = 0u64;
            let mut prev: Option<TcpInfo> = None;
            let mut deadline = start_ns + interval_ns;
            while deadline < end_ns && !stop.load(Ordering::Acquire) {
                sys::sleep_until(deadline);
                let info = tcp_info(&stream)?;
                let secs = interval_ns as f32 / 1e9;
                let (retrans, acked) = prev.as_ref().map_or((0, 0), |p| {
                    (
                        info.total_retrans.wrapping_sub(p.total_retrans),
                        info.bytes_acked.saturating_sub(p.bytes_acked),
                    )
                });
                let x = [
                    info.rtt_us as f32 / 1e3,
                    info.min_rtt_us as f32 / 1e3,
                    info.delivery_rate as f32 * 8.0 / 1e6,
                    info.snd_cwnd as f32,
                    info.unacked as f32,
                    info.lost as f32,
                    retrans as f32,
                    acked as f32 * 8.0 / 1e6 / secs,
                ];
                let (cwnd, pacing) = policy.act(x)?;
                dev.write_all(&action_bytes(cwnd, pacing)?).ctx("writing action")?;
                actions += 1;
                prev = Some(info);
                deadline += interval_ns;
            }
            Ok(actions)
        })
        .ctx("spawning agent")
}
