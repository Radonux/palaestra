//! The path's true state at a step, computed live from the downlink's link
//! records: the seven LCFs a policy may take as its "truth" input (arm A1 of
//! docs/methodology.md, or one LCF patched into net1's beliefs).
//!
//! Each value is defined exactly as the host's labels_at defines it
//! (src/expliccit_env/labels.py), so the truth a policy saw equals the
//! labels its step is stored with; tests/test_truth.py checks that on every
//! step. Every definition is causal: the queue after the last link event at
//! or before the step, and sums over the trailing 100 whole trace ms.
//! Integer sums are kept exact and divided in the host's order, so the
//! values match bit for bit.

use crate::sys::{Result, fail};
use std::collections::VecDeque;

/// policy.TRUTH_COLUMNS, in this order; units as the labels': ns, packets,
/// bit/s, packets, fraction, ns, bit/s.
pub const N: usize = 7;
const WINDOW_MS: u64 = 100;
const WINDOW_S: f64 = 0.1;
const NS_PER_MS: u64 = 1_000_000;
/// One delivery opportunity in bits (Mahimahi's 1504-byte packets).
const OPP_BITS: u64 = 1504 * 8;
const DOWNLINK: u32 = 1;
const MAIN_SPORT: u16 = 5001;
const QUEUE_FULL: u8 = 1;
const ARRIVAL: u16 = 2;
const DROP: u16 = 3;
const DEPART: u16 = 4;
const OPP: u16 = 5;

/// What the qdisc's records say about the downlink, kept only as long as a
/// trailing window can still need it.
pub struct LinkTruth {
    end_ms: u64,
    /// Opportunities per ms over one period, the last entry nonzero
    /// (sch_mmlink's trace; the period is its last index).
    trace: Vec<u8>,
    buffer_pkts: u32,
    one_way_ns: u64,
    /// Arrival instants of accepted packets not yet departed (FIFO).
    queued: VecDeque<u64>,
    /// (ms, queueing delay ns, bytes, observed flow) of each departure.
    departures: VecDeque<(u64, u64, u32, bool)>,
    /// (ms, bytes used) of each processed opportunity millisecond.
    opportunities: VecDeque<(u64, u32)>,
    /// The queue after the latest event, ordered as the host orders ties:
    /// (t_ns, departure 0 < arrival 1 < drop 2, record seq).
    last: Option<((u64, u8, u32), u32)>,
    /// Events after the instant last asked about, not yet eligible.
    later: Vec<((u64, u8, u32), u32)>,
}

impl LinkTruth {
    pub fn new(end_ms: u64, trace: Vec<u8>, buffer_pkts: u32, one_way_ns: u64) -> LinkTruth {
        LinkTruth {
            end_ms,
            trace,
            buffer_pkts,
            one_way_ns,
            queued: VecDeque::new(),
            departures: VecDeque::new(),
            opportunities: VecDeque::new(),
            last: None,
            later: Vec::new(),
        }
    }

    /// Take one framed record (header included); records of other types or
    /// links are ignored. A departure with no accepted arrival left means
    /// records were lost, so the episode fails.
    pub fn observe(&mut self, rec: &[u8]) -> Result<()> {
        let ty = u16_at(rec, 0);
        if !(ARRIVAL..=OPP).contains(&ty) || u32_at(rec, 16) != DOWNLINK {
            return Ok(());
        }
        let (seq, t_ns, ms) = (u32_at(rec, 4), u64_at(rec, 8), u64_at(rec, 24));
        match ty {
            OPP => self.opportunities.push_back((ms, u32_at(rec, 32))),
            ARRIVAL => {
                self.queued.push_back(t_ns);
                self.later.push(((t_ns, 1, seq), u32_at(rec, 40).saturating_add(1)));
            }
            // Only a full queue drops a packet that reached the link; the
            // dropped packet is the arrival just announced.
            DROP if rec.get(62).copied() == Some(QUEUE_FULL) => {
                self.queued.pop_back();
                self.later.push(((t_ns, 2, seq), u32_at(rec, 40)));
            }
            DEPART => {
                let Some(arrived) = self.queued.pop_front() else {
                    return fail(format!("downlink departure (record {seq}) with no accepted arrival"));
                };
                let main = u16_at(rec, 56) == MAIN_SPORT;
                self.departures.push_back((ms, t_ns.saturating_sub(arrived), u32_at(rec, 20), main));
                self.later.push(((t_ns, 0, seq), u32_at(rec, 40)));
            }
            _ => {}
        }
        Ok(())
    }

    /// The seven values at `t_ns`, from every record observed so far;
    /// trace ms 0 starts at `base_ns`.
    #[allow(clippy::cast_precision_loss)]
    pub fn at(&mut self, t_ns: u64, base_ns: u64) -> [f64; N] {
        let mut later = std::mem::take(&mut self.later);
        later.retain(|&(key, q)| {
            if key.0 > t_ns {
                return true;
            }
            if self.last.is_none_or(|(k, _)| key > k) {
                self.last = Some((key, q));
            }
            false
        });
        self.later = later;
        let hi = (t_ns.saturating_sub(base_ns) / NS_PER_MS).min(self.end_ms);
        let lo = hi.saturating_sub(WINDOW_MS);
        while self.departures.front().is_some_and(|d| d.0 < lo) {
            self.departures.pop_front();
        }
        while self.opportunities.front().is_some_and(|o| o.0 < lo) {
            self.opportunities.pop_front();
        }
        let (mut delay_sum, mut n, mut received) = (0u64, 0u64, 0u64);
        for &(_, delay, bytes, main) in self.departures.iter().take_while(|d| d.0 < hi) {
            delay_sum += delay;
            n += 1;
            if main {
                received += u64::from(bytes);
            }
        }
        let used: u64 = self.opportunities.iter().take_while(|o| o.0 < hi).map(|&(_, u)| u64::from(u)).sum();
        let capacity = ((lo..hi).map(|m| self.opportunities_at(m)).sum::<u64>() * OPP_BITS) as f64 / WINDOW_S;
        let queue = self.last.map_or(0, |(_, q)| q);
        [
            if n > 0 { delay_sum as f64 / n as f64 } else { f64::NAN },
            f64::from(queue),
            capacity,
            if self.buffer_pkts > 0 { f64::from(self.buffer_pkts) - f64::from(queue) } else { f64::NAN },
            if capacity > 0.0 { used as f64 * 8.0 / (capacity * WINDOW_S) } else { f64::NAN },
            self.one_way_ns as f64,
            received as f64 * 8.0 / WINDOW_S,
        ]
    }

    /// Opportunities at trace ms `m`, the trace repeating with period P = its
    /// last index (so the last ms of a cycle and ms 0 of the next coincide).
    fn opportunities_at(&self, m: u64) -> u64 {
        let period = self.trace.len().saturating_sub(1) as u64;
        if period == 0 {
            return 0;
        }
        let at =
            |i: u64| usize::try_from(i).ok().and_then(|i| self.trace.get(i)).copied().map_or(0, u64::from);
        let (cycle, r) = (m / period, m % period);
        at(r) + if r == 0 && cycle > 0 { at(period) } else { 0 }
    }
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    b.get(o..o + 2).and_then(|s| s.try_into().ok()).map_or(0, u16::from_le_bytes)
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4).and_then(|s| s.try_into().ok()).map_or(0, u32::from_le_bytes)
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8).and_then(|s| s.try_into().ok()).map_or(0, u64::from_le_bytes)
}
