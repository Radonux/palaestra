//! Sage's control step (contract sage-v1) for one TCP flow: the observed
//! flow, or a competing flow driven by its own policy.
//!
//! Every 10 ms of virtual time the step loop reads TCP_INFO from the sender
//! socket; once the ACK state has moved (bytes acked or packets delivered),
//! it turns the reading into Sage's 77-value state message, writes it to
//! steps.bin and, when a policy runs, asks it for the next action. This is a
//! port of the step in Sage's sage_rl/src/sage.cc (CntThread, commit
//! eedd281), so a policy sees what Sage's actor would. tests/test_sage.py
//! compiles that code, with only the clock and the TCP_INFO read stubbed,
//! and checks the two agree on the same TCP_INFO stream. It passes with a
//! Sage checkout; CI has none and skips it (docs/validation.md).
//!
//! Two message columns are not observations and are written as 0: column 1
//! (Sage's nominal link capacity, taken from its command line) and column 75
//! (Sage's reward, which uses that capacity). The host computes the reward
//! from exact ground truth instead. Column 77 is appended after Sage's 77:
//! the kernel's min RTT in ms, which expliCCit's sage.cc adds to the message
//! (there before the reward; here at the end, so Sage's indices hold).
//!
//! Policy contract (mirrors src/expliccit_env/policy.py), inputs by name:
//!   input  "obs"   float32 [1, 78], the message above
//!   input  "state" float32 [1, S], optional; zeros at the first step
//!   input  "truth" float32 [1, W], optional, W at most truth::N; the first
//!                  W values of the path's true state at the step (truth.rs),
//!                  for oracle policies and LCF patches
//!   input  "cwnd"  float32 [1, 1], optional; the sender's window in packets
//!                  (TCP_INFO's snd_cwnd, the steps table's cwnd column), so
//!                  a policy can set an absolute window as a multiplier
//!   input  "cc_info" float32 [1, 5] under agent_cubic: Orca's statistics
//!                  since the previous step, from TCP_CC_INFO
//!                  (kernel/modules/tcp_agent.c): RTT samples, mean rate
//!                  sample (bytes/s), lost bytes, the period (us), min RTT (us);
//!                  float32 [1, 7] under agent_rate: Aurora's monitor interval
//!                  since the previous step: its length (us), packets sent,
//!                  delivered and lost, RTT samples, their mean (us) and the
//!                  second half's mean less the first's (us); optional
//!   output 0       float32 [1, 1], a: the window is multiplied by 2^a
//!   output 1       float32 [1, S], the next state, when "state" exists
//! a must be finite with |a| <= 8; anything else fails the episode. Under
//! "agent" the window follows Sage: a precise accumulator starting at 10
//! packets, rounded up and kept in [4, 20000]; pacing follows Sage's
//! evaluation setting (pace with the smoothed RTT, coefficient 100),
//! computed here as cwnd * mss / srtt and refreshed every step. Under
//! "agent_pure" the window is the same, and the kernel paces it at every
//! delivery as Sage's kernel does (kernel/modules/tcp_agent.c). Under
//! "agent_cubic" the window is the kernel's current one (the cwnd input)
//! times 2^a, rounded down and kept in [4, 20000], as Orca sets it; a = 0
//! leaves Cubic's window as it is, and the stack paces. Under "agent_rate"
//! a sets a sending rate, from 6 Mbit/s, times 2^a and kept in [0.5, 300]
//! Mbit/s (PCC-RL's deployment), which the kernel paces under a window of
//! 20000; each step lasts half the previous step's mean RTT (cc_info), as
//! Aurora's monitor interval does. A policy may ask for
//! its own step period (Orca's is 20 ms); the message's features keep
//! Sage's definitions whatever the period.
//! ponytail: pacing is refreshed per step, not per ACK as Sage's patched
//! kernel may do (its source is not public); move it into tcp_agent if it
//! matters.

use crate::plan::Mode;
use crate::recorder::Truth;
use crate::sys::{self, Context, Result, fail};
use crate::truth;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::TcpStream;
use std::os::fd::{AsFd, AsRawFd};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

pub const OBS: usize = 78;
/// Linux 7.0's struct tcp_info up to tcpi_reord_seen: the fields Sage
/// reads. tcp_info only grows at its end, so they sit where Sage's 4.19
/// headers put them; their values are this kernel's.
pub const INFO_LEN: usize = 224;
/// t_us, tcp_info, obs, action, cwnd, pad, truth: see src/expliccit_env/steps.py.
pub const RECORD_LEN: usize = 8 + INFO_LEN + 8 * OBS + 8 + 4 + 4 + 8 * truth::N;
const AGENT_ABI: u32 = 2;
const REPORT_PERIOD_US: u64 = 10_000;
/// Sage busy-polls until the ACK state moves; here the loop sleeps this long
/// between polls instead. ponytail: 1 ms resolution after an idle period.
const REPOLL_US: u64 = 1_000;
const BW_NORM_FACTOR: f64 = 100.0;
const CWND_INIT: f64 = 10.0;
const MIN_CWND: u32 = 4;
const MAX_CWND: u32 = 20_000;
const MAX_32BIT: f64 = 2_147_483_647.0; // Sage's MAX_32Bit (0x7FFFFFFF)
const MAX_ACTION: f32 = 8.0;
const PACE_COEF: f64 = 100.0;

/// Sage's dq_sage (define.h): a window of the last `size` values with an
/// incrementally updated mean and a tracked min and max. Its quirks are kept:
/// the min of an empty window, and the floor of every recomputed min, is 1e6;
/// the max starts at 0; avg() of an integer window truncates.
struct Dq<T> {
    dq: VecDeque<T>,
    size: usize,
    average: f64,
    min: Option<T>,
    max: Option<T>,
}

trait Num: Copy + PartialOrd {
    const MIN_INIT: Self;
    const ZERO: Self;
    fn f(self) -> f64;
}
impl Num for f64 {
    const MIN_INIT: f64 = 1e6;
    const ZERO: f64 = 0.0;
    fn f(self) -> f64 {
        self
    }
}
impl Num for u64 {
    const MIN_INIT: u64 = 1_000_000;
    const ZERO: u64 = 0;
    fn f(self) -> f64 {
        self as f64
    }
}

impl<T: Num> Dq<T> {
    fn new(size: usize) -> Dq<T> {
        Dq { dq: VecDeque::with_capacity(size + 1), size, average: 0.0, min: None, max: None }
    }
    fn get_min(&self) -> T {
        self.min.unwrap_or(T::MIN_INIT)
    }
    fn get_max(&self) -> T {
        self.max.unwrap_or(T::ZERO)
    }
    fn get_avg(&self) -> f64 {
        self.average
    }
    fn scan_min(&self) -> T {
        self.dq.iter().fold(T::MIN_INIT, |m, &x| if m > x { x } else { m })
    }
    fn scan_max(&self) -> T {
        self.dq.iter().fold(T::ZERO, |m, &x| if m < x { x } else { m })
    }
    fn add(&mut self, entry: T) {
        let mut new_min = self.get_min();
        let mut new_max = self.get_max();
        let len = self.dq.len();
        if entry < new_min {
            new_min = entry;
        }
        if entry > new_max {
            new_max = entry;
        }
        if len >= self.size {
            let Some(removed) = self.dq.pop_back() else { return };
            self.average = (self.average * len as f64 - removed.f() + entry.f()) / len as f64;
            if removed == self.get_min() {
                new_min = self.scan_min();
                if entry < new_min {
                    new_min = entry;
                }
            }
            if removed == self.get_max() {
                new_max = self.scan_max();
                if entry > new_max {
                    new_max = entry;
                }
            }
        } else {
            self.average =
                if len > 0 { (self.average * len as f64 + entry.f()) / (len + 1) as f64 } else { entry.f() };
        }
        self.min = Some(new_min);
        self.max = Some(new_max);
        self.dq.push_front(entry);
    }
}

impl Dq<u64> {
    fn sum(&self) -> u64 {
        self.dq.iter().fold(0u64, |s, &x| s.wrapping_add(x))
    }
    fn avg(&self) -> u64 {
        if self.dq.is_empty() { 0 } else { self.sum() / self.dq.len() as u64 }
    }
}

/// The TCP_INFO fields Sage uses (offsets from include/uapi/linux/tcp.h).
#[derive(Clone, Copy)]
pub struct Info {
    raw: [u8; INFO_LEN],
}

impl Info {
    fn u8_at(&self, o: usize) -> u8 {
        self.raw.get(o).copied().unwrap_or(0)
    }
    fn u32_at(&self, o: usize) -> u32 {
        self.raw.get(o..o + 4).and_then(|s| s.try_into().ok()).map_or(0, u32::from_ne_bytes)
    }
    fn u64_at(&self, o: usize) -> u64 {
        self.raw.get(o..o + 8).and_then(|s| s.try_into().ok()).map_or(0, u64::from_ne_bytes)
    }
    fn ca_state(&self) -> u8 {
        self.u8_at(1)
    }
    fn rto(&self) -> u32 {
        self.u32_at(8)
    }
    fn ato(&self) -> u32 {
        self.u32_at(12)
    }
    fn snd_mss(&self) -> u32 {
        self.u32_at(16)
    }
    fn unacked(&self) -> u32 {
        self.u32_at(24)
    }
    fn lost(&self) -> u32 {
        self.u32_at(32)
    }
    fn rtt(&self) -> u32 {
        self.u32_at(68)
    }
    fn rttvar(&self) -> u32 {
        self.u32_at(72)
    }
    fn snd_ssthresh(&self) -> u32 {
        self.u32_at(76)
    }
    fn snd_cwnd(&self) -> u32 {
        self.u32_at(80)
    }
    fn pacing_rate(&self) -> u64 {
        self.u64_at(104)
    }
    fn bytes_acked(&self) -> u64 {
        self.u64_at(120)
    }
    fn min_rtt(&self) -> u32 {
        self.u32_at(148)
    }
    fn delivery_rate(&self) -> u64 {
        self.u64_at(160)
    }
    fn delivered(&self) -> u32 {
        self.u32_at(192)
    }
    fn bytes_sent(&self) -> u64 {
        self.u64_at(200)
    }
}

/// The controller's statistics since the last read (TCP_CC_INFO; the read
/// starts a new period): struct agent_cubic_info, five u32, or struct
/// agent_rate_info, a u32, four u16, a u32 and an i32.
fn read_cc_info(stream: &TcpStream, mode: Mode) -> Result<[f64; CC_INFO_MAX]> {
    let mut b = [0u8; 20];
    let n = sys::getsockopt_bytes(stream.as_fd(), libc::IPPROTO_TCP, libc::TCP_CC_INFO, &mut b)
        .ctx("TCP_CC_INFO")?;
    if n != b.len() {
        return fail(format!("TCP_CC_INFO returned {n} bytes, need {}", b.len()));
    }
    let word = |i: usize| -> Option<[u8; 4]> { b.get(i..i + 4).and_then(|w| w.try_into().ok()) };
    let half = |i: usize| -> Option<[u8; 2]> { b.get(i..i + 2).and_then(|w| w.try_into().ok()) };
    let u32_at = |i: usize| -> Result<f64> {
        word(i).map_or_else(|| fail("TCP_CC_INFO field"), |w| Ok(f64::from(u32::from_ne_bytes(w))))
    };
    let u16_at = |i: usize| -> Result<f64> {
        half(i).map_or_else(|| fail("TCP_CC_INFO field"), |w| Ok(f64::from(u16::from_ne_bytes(w))))
    };
    let i32_at = |i: usize| -> Result<f64> {
        word(i).map_or_else(|| fail("TCP_CC_INFO field"), |w| Ok(f64::from(i32::from_ne_bytes(w))))
    };
    let mut out = [0f64; CC_INFO_MAX];
    match mode {
        Mode::Cubic => {
            for (k, v) in out.iter_mut().take(5).enumerate() {
                *v = u32_at(4 * k)?;
            }
        }
        Mode::Rate => {
            out = [u32_at(0)?, u16_at(4)?, u16_at(6)?, u16_at(8)?, u16_at(10)?, u32_at(12)?, i32_at(16)?];
        }
        Mode::Hold | Mode::Pure => return fail("TCP_CC_INFO is read only under agent_cubic or agent_rate"),
    }
    Ok(out)
}

fn read_info(stream: &TcpStream) -> Result<Info> {
    let mut b = [0u8; 256];
    let n =
        sys::getsockopt_bytes(stream.as_fd(), libc::IPPROTO_TCP, libc::TCP_INFO, &mut b).ctx("TCP_INFO")?;
    if n < INFO_LEN {
        return fail(format!("TCP_INFO returned {n} bytes, need {INFO_LEN}"));
    }
    let mut raw = [0u8; INFO_LEN];
    raw.copy_from_slice(&b[..INFO_LEN]);
    Ok(Info { raw })
}

/// Three windows (Sage's SHORT_WIN, MID_WIN, LONG_WIN) of one signal.
struct Smls {
    w: [Dq<f64>; 3],
}

impl Smls {
    fn new() -> Smls {
        Smls { w: [Dq::new(10), Dq::new(200), Dq::new(1000)] }
    }
    fn add(&mut self, x: f64) {
        for d in &mut self.w {
            d.add(x);
        }
    }
    /// avg, min, max of each window into out[first..first + 9].
    fn put(&self, out: &mut [f64; OBS], first: usize) {
        let values = self.w.iter().flat_map(|d| [d.get_avg(), d.get_min(), d.get_max()]);
        for (slot, v) in out.iter_mut().skip(first).zip(values) {
            *slot = v;
        }
    }
}

/// The state CntThread keeps between steps, and the step itself. Only what
/// reaches the message is ported; sage.cc computes more (loss windows,
/// effective-window estimates) that no output depends on.
pub struct Featurizer {
    base_ms: u64,
    t0: u64,
    dt_pre: u64,
    total_bytes_acked_pre: u64,
    pre_pkt_dlv: u64,
    pre_pkt_lost: u64,
    pre_bytes_sent: u64,
    max_sending_rate: u64,
    pre_dr_w_mbps: f64,
    pre_dr_w_max: f64,
    sending_rates: Dq<u64>,
    sent_db: Dq<u64>,
    sent_dt: Dq<u64>,
    dlv_db: Dq<u64>,
    loss_db: Dq<u64>,
    uack_db: Dq<u64>,
    dr_w: Dq<f64>,
    rtt: Smls,
    thr: Smls,
    rtt_rate: Smls,
    rtt_var: Smls,
    inflight: Smls,
    lost: Smls,
}

impl Featurizer {
    /// `start_us`: when the step loop starts (sage.cc's first timestamp());
    /// `base_ms`: trace time 0, for column 0.
    pub fn new(start_us: u64, base_ms: u64) -> Featurizer {
        Featurizer {
            base_ms,
            t0: start_us,
            dt_pre: start_us,
            total_bytes_acked_pre: 0,
            pre_pkt_dlv: 0,
            pre_pkt_lost: 0,
            pre_bytes_sent: 0,
            max_sending_rate: 0,
            pre_dr_w_mbps: 0.0,
            pre_dr_w_max: 1.0,
            sending_rates: Dq::new(5000),
            sent_db: Dq::new(100),
            sent_dt: Dq::new(100),
            dlv_db: Dq::new(100),
            loss_db: Dq::new(100),
            uack_db: Dq::new(100),
            dr_w: Dq::new(200),
            rtt: Smls::new(),
            thr: Smls::new(),
            rtt_rate: Smls::new(),
            rtt_var: Smls::new(),
            inflight: Smls::new(),
            lost: Smls::new(),
        }
    }

    pub fn t0(&self) -> u64 {
        self.t0
    }

    /// One TCP_INFO reading at `now` (us). None if the ACK state has not
    /// moved since the last step; otherwise the message, with column 76 set
    /// from `prev_ratio` (the window multiplier applied since the last step).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    pub fn step(&mut self, s: &Info, now: u64, prev_ratio: f64) -> Option<[f64; OBS]> {
        let bytes_acked_ = s.bytes_acked().wrapping_sub(self.total_bytes_acked_pre) as i32;
        let d_dp = u64::from(s.delivered()).wrapping_sub(self.pre_pkt_dlv);
        let d_db = d_dp.wrapping_mul(u64::from(s.snd_mss()));
        self.pre_pkt_dlv = u64::from(s.delivered());
        if !(bytes_acked_ > 0 || d_db > 0) {
            return None;
        }
        let mss = u64::from(s.snd_mss());
        let dt = match now.wrapping_sub(self.dt_pre) {
            0 => 1,
            d => d,
        };
        self.dt_pre = now;
        let s_db = s.bytes_sent().wrapping_sub(self.pre_bytes_sent);
        let lost = u64::from(s.lost());
        let l_db = if lost > self.pre_pkt_lost { (lost - self.pre_pkt_lost).wrapping_mul(mss) } else { 0 };
        self.pre_pkt_lost = lost;
        self.uack_db.add(u64::from(s.unacked()).wrapping_mul(mss));
        self.sent_db.add(s_db);
        self.dlv_db.add(d_db);
        self.loss_db.add(l_db);
        self.sent_dt.add(dt);
        let dt_sum = self.sent_dt.sum() as f64;
        let dr_w_mbps = 8.0 * self.dlv_db.sum() as f64 / dt_sum;
        let l_w_mbps = 8.0 * self.loss_db.sum() as f64 / dt_sum;
        self.dr_w.add(dr_w_mbps);
        let mut dr_w_max = self.dr_w.scan_max();
        if dr_w_max == 0.0 {
            dr_w_max = 1.0;
        }
        let mut time_delta = (now as f64 - self.t0 as f64) / 1_000_000.0;
        self.total_bytes_acked_pre = s.bytes_acked();
        let mut acked_rate = f64::from(bytes_acked_) / time_delta;
        let mut sending_rate = s.bytes_sent().wrapping_sub(self.pre_bytes_sent) as f64;
        self.pre_bytes_sent = s.bytes_sent();
        let max_delivary_rate = (dr_w_max * 1e6 / 8.0) as u64;
        let min_rtt_us = u64::from(s.min_rtt());
        if max_delivary_rate == 0 {
            acked_rate = 0.0;
        }
        let rtt_rate;
        if s.rtt() > 0 {
            sending_rate /= time_delta;
            self.sending_rates.add(sending_rate as u64);
            self.max_sending_rate = self.sending_rates.scan_max();
            rtt_rate = min_rtt_us as f64 / f64::from(s.rtt());
            time_delta = (1_000_000.0 * time_delta) / f64::from(s.min_rtt());
            acked_rate /= self.max_sending_rate as f64;
        } else {
            time_delta = (REPORT_PERIOD_US / 1000) as f64;
            rtt_rate = 0.0;
        }
        let s_db_tmp = self.sent_db.sum();
        let ua_db_tmp = self.uack_db.avg();
        let cwnd_unacked_rate =
            if s_db_tmp > 0 { ua_db_tmp as f64 / s_db_tmp as f64 } else { ua_db_tmp as f64 };
        let cwnd_bits = match u64::from(s.snd_cwnd()).wrapping_mul(mss).wrapping_mul(8) {
            0 => 1,
            b => b,
        };

        self.rtt.add(f64::from(s.rtt()) / 100_000.0);
        self.rtt_rate.add(rtt_rate);
        self.rtt_var.add(f64::from(s.rttvar()) / 1000.0);
        self.thr.add(s.delivery_rate() as f64 / 125_000.0 / BW_NORM_FACTOR);
        self.inflight.add(f64::from(s.unacked()) / 1000.0);
        self.lost.add(f64::from(s.lost()) / 100.0);

        let mut m = [0.0f64; OBS];
        m[0] = f64::from(((now / 1000) as i64 - self.base_ms as i64) as i32) / 1e3;
        m[2] = f64::from(s.rtt()) / 100_000.0;
        m[3] = f64::from(s.rttvar()) / 1000.0;
        m[4] = f64::from(s.rto()) / 100_000.0;
        m[5] = f64::from(s.ato()) / 100_000.0;
        m[6] = s.pacing_rate() as f64 / 125_000.0 / BW_NORM_FACTOR;
        m[7] = s.delivery_rate() as f64 / 125_000.0 / BW_NORM_FACTOR;
        m[8] = f64::from(s.snd_ssthresh() as i32); // printed with %d
        m[9] = f64::from(s.ca_state());
        self.rtt.put(&mut m, 10);
        self.thr.put(&mut m, 19);
        self.rtt_rate.put(&mut m, 28);
        self.rtt_var.put(&mut m, 37);
        self.inflight.put(&mut m, 46);
        self.lost.put(&mut m, 55);
        m[64] = dr_w_mbps - l_w_mbps;
        m[65] = time_delta;
        m[66] = rtt_rate;
        m[67] = l_w_mbps / BW_NORM_FACTOR;
        m[68] = acked_rate;
        m[69] = if self.pre_dr_w_mbps > 0.0 { dr_w_mbps / self.pre_dr_w_mbps } else { dr_w_mbps };
        m[70] = (dr_w_max * f64::from(s.min_rtt())) / cwnd_bits as f64;
        m[71] = dr_w_mbps / BW_NORM_FACTOR;
        m[72] = cwnd_unacked_rate;
        m[73] = if self.pre_dr_w_max > 0.0 { dr_w_max / self.pre_dr_w_max } else { dr_w_max };
        m[74] = dr_w_max / BW_NORM_FACTOR;
        m[76] = prev_action(prev_ratio);
        m[77] = f64::from(s.min_rtt()) / 1000.0;
        self.t0 = now;
        self.pre_dr_w_mbps = dr_w_mbps;
        self.pre_dr_w_max = dr_w_max;
        Some(m)
    }
}

/// Column 76: round(log2f(ratio) * 1000) / 1000 in float, as sage.cc prints it.
#[allow(clippy::cast_possible_truncation)]
fn prev_action(ratio: f64) -> f64 {
    if ratio > 0.0 {
        f64::from(((ratio as f32).log2() * 1000.0).round()) / 1000.0
    } else {
        f64::from(0.0001f32.log2())
    }
}

/// SplitMix64 (Steele, Lea and Flood, 2014): a small seeded generator for
/// exploration noise, the same stream on every run with the same seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    #[allow(clippy::cast_precision_loss)]
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The inputs a policy may declare, in the graph's order.
enum Input {
    Obs,
    State,
    /// The first so many values of the true state.
    Truth(usize),
    Cwnd,
    CcInfo,
}

/// TCP_CC_INFO as policy input: 5 values under agent_cubic, 7 under agent_rate.
pub const CC_INFO_MAX: usize = 7;
/// agent_rate's starting rate and bounds, bits/s (PCC-RL's loaded_client.py).
const RATE_INIT_BPS: f64 = 6e6;
const RATE_MIN_BPS: f64 = 0.5e6;
const RATE_MAX_BPS: f64 = 300e6;
/// agent_rate's shortest step, us.
const RATE_MIN_PERIOD_US: u64 = 1_000;

/// An ONNX policy under contract sage-v1, with optional exploration noise.
pub struct Policy {
    plan: TypedRunnableModel<TypedModel>,
    mode: Mode,
    period_us: u64,
    inputs: Vec<Input>,
    state: Option<Tensor>,
    sigma: f64,
    rng: SplitMix64,
    hold: u64,
    held: f64,
}

/// S of input `i` declared with the fixed shape [1, S].
fn fixed_width(model: &InferenceModel, i: usize) -> Result<Option<usize>> {
    let fact = model.input_fact(i).ctx("policy input")?;
    let dims: Option<Vec<usize>> = fact
        .shape
        .dims()
        .map(|d| d.concretize().and_then(|d| d.to_i64().ok()).and_then(|d| usize::try_from(d).ok()))
        .collect();
    Ok(match dims.as_deref() {
        Some([1, s]) => Some(*s),
        _ => None,
    })
}

impl Policy {
    pub fn load(path: &Path, noise: crate::plan::Noise, mode: Mode, period_ms: u32) -> Result<Policy> {
        let mut model =
            tract_onnx::onnx().model_for_path(path).ctx(format!("loading ONNX policy {}", path.display()))?;
        let names: Vec<String> = model.inputs.iter().map(|o| model.node(o.node).name.clone()).collect();
        let (mut inputs, mut state) = (Vec::with_capacity(names.len()), None);
        for (i, name) in names.iter().enumerate() {
            let (input, width) = match name.as_str() {
                "obs" => (Input::Obs, OBS),
                "truth" => match fixed_width(&model, i)? {
                    Some(w) if (1..=truth::N).contains(&w) => (Input::Truth(w), w),
                    _ => {
                        return fail(format!(
                            "policy truth input must have a fixed shape [1, 1 to {}]",
                            truth::N
                        ));
                    }
                },
                "cwnd" => (Input::Cwnd, 1),
                "cc_info" => match mode {
                    Mode::Cubic => (Input::CcInfo, 5),
                    Mode::Rate => (Input::CcInfo, CC_INFO_MAX),
                    Mode::Hold | Mode::Pure => {
                        return fail("policy input cc_info needs cc agent_cubic or agent_rate");
                    }
                },
                "state" => match fixed_width(&model, i)? {
                    Some(s) => {
                        state = Some(Tensor::zero::<f32>(&[1, s]).ctx("state")?);
                        (Input::State, s)
                    }
                    None => return fail("policy state input must have a fixed shape [1, S]"),
                },
                other => {
                    return fail(format!("policy input {other:?} is not obs, state, truth, cwnd or cc_info"));
                }
            };
            model = model.with_input_fact(i, f32::fact([1, width]).into()).ctx(format!("{name} input"))?;
            inputs.push(input);
        }
        let count = |want: fn(&Input) -> bool| inputs.iter().filter(|i| want(i)).count();
        if count(|i| matches!(i, Input::Obs)) != 1
            || count(|i| matches!(i, Input::State)) > 1
            || count(|i| matches!(i, Input::Truth(_))) > 1
            || count(|i| matches!(i, Input::Cwnd)) > 1
            || count(|i| matches!(i, Input::CcInfo)) > 1
        {
            return fail(format!(
                "policy inputs {names:?}: need one obs, at most one state, truth, cwnd and cc_info"
            ));
        }
        let plan = model
            .into_optimized()
            .and_then(|m| m.into_runnable())
            .ctx(format!("optimizing ONNX policy {}", path.display()))?;
        Ok(Policy {
            plan,
            mode,
            period_us: u64::from(period_ms) * 1000,
            inputs,
            state,
            sigma: noise.sigma,
            rng: SplitMix64(noise.seed),
            hold: u64::from(noise.hold.get()),
            held: 0.0,
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    /// Whether the policy reads TCP_CC_INFO (read only then: reading starts
    /// a new period).
    fn reads_cc_info(&self) -> bool {
        self.inputs.iter().any(|i| matches!(i, Input::CcInfo))
    }

    fn act(
        &mut self,
        obs: &[f64; OBS],
        truth: &[f64; truth::N],
        cwnd: u32,
        cc: &[f64; CC_INFO_MAX],
    ) -> Result<f32> {
        let row = |v: &[f64]| -> Result<TValue> {
            let x: Vec<f32> = v.iter().map(|&v| v as f32).collect();
            let t: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, v.len()), x).ctx("policy input")?.into();
            Ok(t.into())
        };
        let mut inputs = TVec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            inputs.push(match input {
                Input::Obs => row(obs)?,
                Input::Truth(w) => row(truth.get(..*w).unwrap_or(truth))?,
                Input::Cwnd => row(&[f64::from(cwnd)])?,
                Input::CcInfo => row(if matches!(self.mode, Mode::Cubic) { &cc[..5] } else { cc })?,
                Input::State => match self.state.take() {
                    Some(s) => s.into(),
                    None => return fail("policy state was not returned by the previous step"),
                },
            });
        }
        let mut out = self.plan.run(inputs).ctx("running policy")?;
        if out.is_empty() {
            return fail("policy returned no output");
        }
        let action = out.remove(0);
        let v: Vec<f32> = action.to_array_view::<f32>().ctx("policy output type")?.iter().copied().collect();
        let stateful = self.inputs.iter().any(|i| matches!(i, Input::State));
        match out.first() {
            Some(s) if stateful => self.state = Some(s.clone().into_tensor()),
            None if stateful => return fail("policy has a state input but no state output"),
            _ => {}
        }
        match v.as_slice() {
            [a] if a.is_finite() && a.abs() <= MAX_ACTION => Ok(*a),
            [a] => fail(format!("policy action {a} is not finite or exceeds {MAX_ACTION}")),
            _ => fail(format!("policy output has {} values, expected 1", v.len())),
        }
    }
}

/// Sage's window accumulator (ACCUMULATE_CWND, ROUNT_TYPE CEILING).
struct Window {
    precise: f64,
}

impl Window {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply(&mut self, ratio: f64) -> u32 {
        let product = self.precise * ratio;
        self.precise = if product > MAX_32BIT { MAX_32BIT } else { product };
        let mut cwnd = self.precise.ceil().min(MAX_32BIT) as u32;
        if cwnd < MIN_CWND {
            cwnd = MIN_CWND;
            self.precise = f64::from(MIN_CWND);
        }
        if cwnd > MAX_CWND {
            cwnd = MAX_CWND;
            self.precise = f64::from(MAX_CWND);
        }
        cwnd
    }
}

/// Orca's window: the kernel's current one times `ratio`, rounded down
/// (Orca's integer arithmetic), kept in [4, 20000]. Orca's ratios are whole
/// percents, so the product is a multiple of 0.01; the margin absorbs the
/// float32 error of a = log2(ratio), at most 0.0035 at the largest window.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn on_kernel_window(ratio: f64, snd_cwnd: u32) -> u32 {
    let w = (ratio * f64::from(snd_cwnd) + 5e-3).floor();
    (w.clamp(f64::from(MIN_CWND), f64::from(MAX_CWND))) as u32
}

/// Aurora's sending rate (PCC-RL's loaded_client.py): from 6 Mbit/s, each
/// action multiplies it, and it is kept in [0.5, 300] Mbit/s.
struct Rate {
    bps: f64,
}

impl Rate {
    /// The new rate, bytes per second.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn apply(&mut self, ratio: f64) -> u64 {
        self.bps = (self.bps * ratio).clamp(RATE_MIN_BPS, RATE_MAX_BPS);
        (self.bps / 8.0) as u64
    }
}

/// The pacing rate an action carries: Sage's (cwnd * mss / srtt, unpaced
/// until the first RTT sample) under agent, Aurora's rate under agent_rate,
/// none where the kernel paces (agent_pure, agent_cubic).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn pacing_of(mode: Mode, cwnd: u32, info: &Info, rate: u64) -> u64 {
    match (mode, info.rtt()) {
        (Mode::Pure | Mode::Cubic, _) | (Mode::Hold, 0) => 0,
        (Mode::Rate, _) => rate,
        (Mode::Hold, srtt) => {
            (f64::from(cwnd) * f64::from(info.snd_mss()) * 1e6 / f64::from(srtt) * PACE_COEF / 100.0) as u64
        }
    }
}

/// struct expl_agent_action (kernel/modules/expliccit_rec.h) for the socket
/// `fd` of this process.
fn agent_action(fd: i32, cwnd: u32, pacing: u64) -> [u8; 24] {
    let mut b = [0u8; 24];
    b[0..4].copy_from_slice(&AGENT_ABI.to_ne_bytes());
    b[4..8].copy_from_slice(&fd.to_ne_bytes());
    b[8..12].copy_from_slice(&cwnd.to_ne_bytes());
    b[16..24].copy_from_slice(&pacing.to_ne_bytes());
    b
}

pub struct Steps {
    pub count: u64,
    pub actions: u64,
}

/// One flow's step loop, set up before its thread starts so that a file or
/// device that cannot be opened fails the episode at once.
pub struct Loop {
    file: File,
    dev: Option<File>,
    policy: Option<Policy>,
    truth: Truth,
    /// The flow's sender port, which its truth input is computed for.
    sport: u16,
    start_ns: u64,
    base_ns: u64,
    end_ns: u64,
    stop: Arc<AtomicBool>,
    /// Let the kernel finish the step's instant before reading the truth.
    settle: bool,
}

impl Loop {
    /// Steps from `start_ns` until `end_ns` (or `stop`), each written to
    /// `out` and, with a policy, acted on.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        out: &Path,
        policy: Option<Policy>,
        truth: Truth,
        sport: u16,
        start_ns: u64,
        base_ns: u64,
        end_ns: u64,
        stop: Arc<AtomicBool>,
        settle: bool,
    ) -> Result<Loop> {
        let file = File::create(out).ctx(format!("creating {}", out.display()))?;
        let dev = match policy {
            Some(_) => {
                Some(File::options().write(true).open("/dev/expl_agent").ctx("opening /dev/expl_agent")?)
            }
            None => None,
        };
        Ok(Loop { file, dev, policy, truth, sport, start_ns, base_ns, end_ns, stop, settle })
    }

    /// Run the loop on `stream`, the flow's sender socket.
    pub fn run(self, stream: &TcpStream) -> Result<Steps> {
        let Loop { file, mut dev, mut policy, truth, sport, start_ns, base_ns, end_ns, stop, settle } = self;
        let fd = stream.as_raw_fd();
        {
            let mut w = BufWriter::new(file);
            let start_us = start_ns / 1000;
            let end_us = end_ns / 1000;
            let mut feat = Featurizer::new(start_us, base_ns / 1_000_000);
            let mut window = Window { precise: CWND_INIT };
            let mut rate = Rate { bps: RATE_INIT_BPS };
            // The multiplier applied since the last step: the policy's, or
            // for any other controller the observed change of the window
            // (Sage's collector logs cwnd / previous cwnd, starting from 10).
            let mut ratio = 1.0f64;
            let mut pre_cwnd = CWND_INIT;
            let (mut count, mut actions) = (0u64, 0u64);
            let mut period = policy.as_ref().map_or(REPORT_PERIOD_US, |p| p.period_us);
            let mut poll = start_us + period;
            while poll < end_us && !stop.load(Ordering::Acquire) {
                sys::sleep_until(poll * 1000);
                let info = read_info(stream)?;
                let now_ns = sys::monotonic_ns();
                let now = now_ns / 1000;
                // With several step loops, softirq work at this instant can
                // overflow to ksoftirqd and run after this thread, so its
                // link records would be drained late. Sleeping 1 ns lets
                // the instant finish first; the truth is still taken at
                // `now`. One loop never needs it (docs/validation.md), and
                // skipping it keeps single-agent episodes as they were.
                if settle {
                    sys::sleep_until(now_ns + 1);
                }
                if policy.is_none() {
                    ratio = f64::from(info.snd_cwnd()) / pre_cwnd;
                }
                let Some(obs) = feat.step(&info, now, ratio) else {
                    poll = now + REPOLL_US;
                    continue;
                };
                if policy.is_none() {
                    pre_cwnd = f64::from(info.snd_cwnd());
                }
                let true_state = truth.at(now * 1000, base_ns, sport)?;
                let (mut a, mut cwnd) = (f64::NAN, 0u32);
                if let (Some(p), Some(dev)) = (policy.as_mut(), dev.as_mut()) {
                    let cc =
                        if p.reads_cc_info() { read_cc_info(stream, p.mode)? } else { [0.0; CC_INFO_MAX] };
                    let act = f64::from(p.act(&obs, &true_state, info.snd_cwnd(), &cc)?);
                    // The noise is drawn every step whatever sigma and hold
                    // are, so the stream does not depend on them; each
                    // block of `hold` actions keeps its first draw. The
                    // applied action is recorded.
                    let z = p.rng.normal();
                    if actions % p.hold == 0 {
                        p.held = z;
                    }
                    a = (act + p.sigma * p.held).clamp(-f64::from(MAX_ACTION), f64::from(MAX_ACTION));
                    ratio = 2f64.powf(a);
                    let mut bytes_per_s = 0;
                    cwnd = match p.mode {
                        Mode::Hold | Mode::Pure => window.apply(ratio),
                        Mode::Cubic => on_kernel_window(ratio, info.snd_cwnd()),
                        Mode::Rate => {
                            bytes_per_s = rate.apply(ratio);
                            // Aurora's next monitor interval: half this one's mean RTT.
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            if cc[5] > 0.0 {
                                period = RATE_MIN_PERIOD_US.max((cc[5] / 2.0) as u64);
                            }
                            MAX_CWND
                        }
                    };
                    let pacing = pacing_of(p.mode, cwnd, &info, bytes_per_s);
                    dev.write_all(&agent_action(fd, cwnd, pacing)).ctx("writing action")?;
                    actions += 1;
                }
                let mut rec = Vec::with_capacity(RECORD_LEN);
                rec.extend_from_slice(&now.to_le_bytes());
                rec.extend_from_slice(&info.raw);
                obs.iter().for_each(|v| rec.extend_from_slice(&v.to_le_bytes()));
                rec.extend_from_slice(&a.to_le_bytes());
                rec.extend_from_slice(&cwnd.to_le_bytes());
                rec.extend_from_slice(&0u32.to_le_bytes());
                true_state.iter().for_each(|v| rec.extend_from_slice(&v.to_le_bytes()));
                if rec.len() != RECORD_LEN {
                    return fail(format!("step record is {} bytes, expected {RECORD_LEN}", rec.len()));
                }
                w.write_all(&rec).ctx("steps.bin")?;
                count += 1;
                poll = feat.t0() + period;
            }
            w.flush().ctx("steps.bin")?;
            Ok(Steps { count, actions })
        }
    }
}

/// Run `lp` on `stream` in a thread named `name`.
pub fn spawn(name: &str, lp: Loop, stream: Arc<TcpStream>) -> Result<JoinHandle<Result<Steps>>> {
    std::thread::Builder::new().name(name.into()).spawn(move || lp.run(&stream)).ctx("spawning step loop")
}
