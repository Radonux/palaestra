//! The path's true state at a step, computed live from the links' records:
//! every label the host stores that is known at the step, which a policy
//! may take as its "truth" input (arm A1 of docs/methodology.md, or one LCF
//! patched into net1's beliefs).
//!
//! Each value is defined exactly as the host's labels_at defines it
//! (src/expliccit_env/labels.py), so the truth a policy saw equals the
//! labels its step is stored with; tests/test_truth.py checks that on every
//! step and column. The one stored label left out is queue_delay_ns, the
//! delay of a packet that may not have left yet. Every other definition is
//! causal (the queue after the last link event at or before the step, sums
//! over trailing whole trace ms) or reads the trace ahead, which the guest
//! holds (capacity_ahead_bps). Integer sums are kept exact and divided in
//! the host's order, so the values match bit for bit.

use crate::sys::{Result, fail};
use std::collections::VecDeque;

/// policy.TRUTH_COLUMNS, in this order and in the labels' units:
/// expliCCit's seven first (a policy whose truth input is narrower reads a
/// prefix), then the other instant labels, then the other windowed labels
/// over 100 ms, then every windowed label over 10 ms.
pub const N: usize = 36;
/// labels.WINDOWS_MS and the number of labels.WINDOWED.
const WINDOWS_MS: [u64; 2] = [100, 10];
const WINDOWED: usize = 13;
const _: () = assert!(N == 14 + (WINDOWED - 4) + WINDOWED);
const NS_PER_MS: u64 = 1_000_000;
/// One delivery opportunity in bits (Mahimahi's 1504-byte packets).
const OPP_BITS: u64 = 1504 * 8;
const DOWNLINK: u32 = 1;
const UPLINK: u32 = 2;
const QUEUE_FULL: u8 = 1;
const RANDOM: u8 = 5;
const ARRIVAL: u16 = 2;
const DROP: u16 = 3;
const DEPART: u16 = 4;
const OPP: u16 = 5;

/// What the path is configured with (the host's labels.PathConstants).
pub struct Path {
    pub end_ms: u64,
    /// Opportunities per ms over one period, the last entry nonzero
    /// (sch_mmlink's trace; the period is its last index).
    pub trace: Vec<u8>,
    /// Droptail limits, 0 for none.
    pub buffer_pkts: u32,
    pub buffer_bytes: u32,
    pub one_way_ns: u64,
    /// The data direction's i.i.d. drop probability times 2^32.
    pub loss: u32,
    /// (sender port, extra base RTT) of flows with an RTT of their own: their
    /// one-way delay is the path's plus half the extra, as the host labels it.
    pub extra_rtt_ns: Vec<(u16, u64)>,
}

/// A change of the bottleneck queue: when (ordered as the host orders ties:
/// t_ns, departure 0 < arrival 1 < drop 2, record seq), the queue after it,
/// and the packet's sender port.
struct Event {
    key: (u64, u8, u32),
    pkts: u32,
    bytes: u32,
    sport: u16,
}

/// The queue over one trace ms that had an event: its least value and its
/// value after the ms's last event.
struct MsQueue {
    ms: i64,
    least: u32,
    key: (u64, u8, u32),
    end: u32,
}

/// What the qdiscs' records say about the path, kept only as long as a
/// trailing window can still need it.
pub struct LinkTruth {
    path: Path,
    /// Arrival instants of accepted packets not yet departed (FIFO), for the
    /// bottleneck and for the ACK direction's link.
    queued: VecDeque<u64>,
    ack_queued: VecDeque<u64>,
    /// (ms, queueing delay ns, bytes, sender port) of each departure.
    departures: VecDeque<(u64, u64, u32, u16)>,
    /// (ms, queueing delay ns) of each departure of the ACK direction's link.
    ack_departures: VecDeque<(u64, u64)>,
    /// (ms, bytes used) of each processed opportunity millisecond.
    opportunities: VecDeque<(u64, u32)>,
    /// (ms, bytes, sender port) of each packet that reached the bottleneck.
    arrivals: VecDeque<(u64, u32, u16)>,
    /// (ms, sender port, by the full queue: else by the random loss) of
    /// each packet those two dropped.
    drops: VecDeque<(u64, u16, bool)>,
    /// Queue events after the instant last asked about, in record order.
    later: Vec<Event>,
    /// The state the events up to that instant leave: the queue after the
    /// latest, the senders of the packets that arrived and have not departed
    /// (FIFO), and the queue per ms.
    last: Option<Event>,
    waiting: VecDeque<u16>,
    per_ms: VecDeque<MsQueue>,
}

impl LinkTruth {
    pub fn new(path: Path) -> LinkTruth {
        LinkTruth {
            path,
            queued: VecDeque::new(),
            ack_queued: VecDeque::new(),
            departures: VecDeque::new(),
            ack_departures: VecDeque::new(),
            opportunities: VecDeque::new(),
            arrivals: VecDeque::new(),
            drops: VecDeque::new(),
            later: Vec::new(),
            last: None,
            waiting: VecDeque::new(),
            per_ms: VecDeque::new(),
        }
    }

    /// Take one framed record (header included); records of other types or
    /// links are ignored. A departure with no accepted arrival left means
    /// records were lost, so the episode fails.
    pub fn observe(&mut self, rec: &[u8]) -> Result<()> {
        let ty = u16_at(rec, 0);
        if !(ARRIVAL..=OPP).contains(&ty) {
            return Ok(());
        }
        let (seq, t_ns, ms) = (u32_at(rec, 4), u64_at(rec, 8), u64_at(rec, 24));
        let queue_full = rec.get(62).copied() == Some(QUEUE_FULL);
        let lost = || fail(format!("link departure (record {seq}) with no accepted arrival"));
        match (u32_at(rec, 16), ty) {
            (UPLINK, ARRIVAL) => self.ack_queued.push_back(t_ns),
            (UPLINK, DROP) if queue_full => {
                self.ack_queued.pop_back();
            }
            (UPLINK, DEPART) => {
                let Some(arrived) = self.ack_queued.pop_front() else { return lost() };
                self.ack_departures.push_back((ms, t_ns.saturating_sub(arrived)));
            }
            (DOWNLINK, OPP) => self.opportunities.push_back((ms, u32_at(rec, 32))),
            (DOWNLINK, _) => {
                let (bytes, sport) = (u32_at(rec, 20), u16_at(rec, 56));
                let (pkts, queue_bytes) = (u32_at(rec, 40), u32_at(rec, 44));
                let mut event = |rank, pkts, bytes| {
                    self.later.push(Event { key: (t_ns, rank, seq), pkts, bytes, sport });
                };
                match ty {
                    ARRIVAL => {
                        event(1, pkts.saturating_add(1), queue_bytes.saturating_add(bytes));
                        self.queued.push_back(t_ns);
                        self.arrivals.push_back((ms, bytes, sport));
                    }
                    // Only a full queue drops a packet that reached the
                    // link; the dropped packet is the arrival just announced.
                    DROP if queue_full => {
                        event(2, pkts, queue_bytes);
                        self.queued.pop_back();
                        self.drops.push_back((ms, sport, true));
                    }
                    DROP if rec.get(62).copied() == Some(RANDOM) => {
                        self.drops.push_back((ms, sport, false));
                    }
                    DEPART => {
                        event(0, pkts, queue_bytes);
                        let Some(arrived) = self.queued.pop_front() else { return lost() };
                        self.departures.push_back((ms, t_ns.saturating_sub(arrived), bytes, sport));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Apply the queue events up to `t_ns`, in record order.
    fn advance(&mut self, t_ns: u64, base_ns: u64) -> Result<()> {
        for e in std::mem::take(&mut self.later) {
            if e.key.0 > t_ns {
                self.later.push(e);
                continue;
            }
            match e.key.1 {
                0 => {
                    if self.waiting.pop_front().is_none() {
                        return fail(format!("link departure (record {}) with nothing waiting", e.key.2));
                    }
                }
                1 => self.waiting.push_back(e.sport),
                _ => {
                    self.waiting.pop_back();
                }
            }
            let ms = (i128::from(e.key.0) - i128::from(base_ns)).div_euclid(i128::from(NS_PER_MS));
            let ms = i64::try_from(ms).unwrap_or(i64::MAX);
            let at = self.per_ms.partition_point(|m| m.ms < ms);
            match self.per_ms.get_mut(at).filter(|m| m.ms == ms) {
                Some(m) => {
                    m.least = m.least.min(e.pkts);
                    if e.key > m.key {
                        (m.key, m.end) = (e.key, e.pkts);
                    }
                }
                None => self.per_ms.insert(at, MsQueue { ms, least: e.pkts, key: e.key, end: e.pkts }),
            }
            if self.last.as_ref().is_none_or(|l| e.key > l.key) {
                self.last = Some(e);
            }
        }
        Ok(())
    }

    /// Every value at `t_ns`, from every record observed so far, for the
    /// flow with sender port `sport` (its rates, drops, queue share and
    /// delay; the others are the path's); trace ms 0 starts at `base_ns`.
    #[allow(clippy::cast_precision_loss)]
    pub fn at(&mut self, t_ns: u64, base_ns: u64, sport: u16) -> Result<[f64; N]> {
        self.advance(t_ns, base_ns)?;
        let hi = (t_ns.saturating_sub(base_ns) / NS_PER_MS).min(self.path.end_ms);
        self.forget(hi.saturating_sub(WINDOWS_MS.into_iter().max().unwrap_or(0)));
        let (queue, queue_bytes) = self.last.as_ref().map_or((0, 0), |l| (l.pkts, l.bytes));
        // The packet in transit has left the queue but not departed: it is
        // the head of those waiting when they are more than the queue.
        let in_transit = self.waiting.len() > queue as usize && self.waiting.front() == Some(&sport);
        let flow_queue = self.waiting.iter().filter(|&&p| p == sport).count() - usize::from(in_transit);
        let p = &self.path;
        let fill = |q: u32, limit: u32| (limit > 0).then(|| f64::from(q) / f64::from(limit));
        let one_way = p.one_way_ns + p.extra_rtt_ns.iter().find(|e| e.0 == sport).map_or(0, |e| e.1 / 2);
        let [wide, narrow] = WINDOWS_MS.map(|w| self.window(w, hi, sport));
        let [capacity, ahead, delay, ack_delay, least, busy, receive, rest @ ..] = wide;
        let mut out = [0.0; N];
        let values = [
            delay,
            f64::from(queue),
            capacity,
            if p.buffer_pkts > 0 { f64::from(p.buffer_pkts) - f64::from(queue) } else { f64::NAN },
            busy,
            one_way as f64,
            receive,
            f64::from(queue_bytes),
            flow_queue as f64,
            match (fill(queue, p.buffer_pkts), fill(queue_bytes, p.buffer_bytes)) {
                (Some(a), Some(b)) => a.max(b),
                (a, b) => a.or(b).unwrap_or(f64::NAN),
            },
            (2 * one_way) as f64,
            f64::from(p.buffer_pkts),
            f64::from(p.buffer_bytes),
            f64::from(p.loss) / 4_294_967_296.0,
            ahead,
            ack_delay,
            least,
        ];
        for (o, v) in out.iter_mut().zip(values.into_iter().chain(rest).chain(narrow)) {
            *o = v;
        }
        Ok(out)
    }

    /// Drop what no window that starts at trace ms `lo` or later can need.
    fn forget(&mut self, lo: u64) {
        while self.departures.front().is_some_and(|d| d.0 < lo) {
            self.departures.pop_front();
        }
        while self.ack_departures.front().is_some_and(|d| d.0 < lo) {
            self.ack_departures.pop_front();
        }
        while self.opportunities.front().is_some_and(|o| o.0 < lo) {
            self.opportunities.pop_front();
        }
        while self.arrivals.front().is_some_and(|a| a.0 < lo) {
            self.arrivals.pop_front();
        }
        while self.drops.front().is_some_and(|d| d.0 < lo) {
            self.drops.pop_front();
        }
        // The last ms before the window says what queue the window starts with.
        let lo = i64::try_from(lo).unwrap_or(i64::MAX);
        while self.per_ms.get(1).is_some_and(|m| m.ms < lo) {
            self.per_ms.pop_front();
        }
    }

    /// labels.WINDOWED, in that order, over the `w` whole trace ms before `hi`.
    #[allow(clippy::cast_precision_loss)]
    fn window(&self, w: u64, hi: u64, sport: u16) -> [f64; WINDOWED] {
        let lo = hi.saturating_sub(w);
        let inside = |ms: u64| (lo..hi).contains(&ms);
        let window_s = w as f64 / 1000.0;
        let rate = |bytes: u64| bytes as f64 * 8.0 / window_s;
        let ratio = |a: u64, b: u64| if b > 0 { a as f64 / b as f64 } else { f64::NAN };
        let opportunities = |from: u64| (from..from + w).map(|m| self.opportunities_at(m)).sum::<u64>();
        let capacity = ((lo..hi).map(|m| self.opportunities_at(m)).sum::<u64>() * OPP_BITS) as f64 / window_s;

        let (mut delay_sum, mut departed, mut received, mut cross) = (0u64, 0u64, 0u64, 0u64);
        for &(_, delay, bytes, port) in self.departures.iter().filter(|d| inside(d.0)) {
            delay_sum += delay;
            departed += 1;
            *(if port == sport { &mut received } else { &mut cross }) += u64::from(bytes);
        }
        let (mut ack_sum, mut ack_n) = (0u64, 0u64);
        for &(_, delay) in self.ack_departures.iter().filter(|d| inside(d.0)) {
            ack_sum += delay;
            ack_n += 1;
        }
        let used: u64 = self.opportunities.iter().filter(|o| inside(o.0)).map(|&(_, u)| u64::from(u)).sum();
        let (mut arrived, mut own_bytes, mut own) = (0u64, 0u64, 0u64);
        let mut flows: Vec<u16> = Vec::new();
        for &(_, bytes, port) in self.arrivals.iter().filter(|a| inside(a.0)) {
            arrived += u64::from(bytes);
            if port == sport {
                own_bytes += u64::from(bytes);
                own += 1;
            }
            if !flows.contains(&port) {
                flows.push(port);
            }
        }
        let dropped = |full: bool| {
            self.drops.iter().filter(|d| inside(d.0) && d.1 == sport && d.2 == full).count() as u64
        };
        let random = dropped(false);
        // The queue the window starts with, then the least any event left.
        let (lo_ms, hi_ms) = (i64::try_from(lo).unwrap_or(i64::MAX), i64::try_from(hi).unwrap_or(i64::MAX));
        let start = self.per_ms.iter().take_while(|m| m.ms < lo_ms).last().map_or(0, |m| m.end);
        let least = self
            .per_ms
            .iter()
            .filter(|m| (lo_ms..hi_ms).contains(&m.ms))
            .map(|m| m.least)
            .fold(start, u32::min);
        let offered = capacity * window_s / 8.0;
        [
            capacity,
            (opportunities(hi) * OPP_BITS) as f64 / window_s,
            ratio(delay_sum, departed),
            ratio(ack_sum, ack_n),
            if lo < hi { f64::from(least) } else { f64::NAN },
            if capacity > 0.0 { used as f64 * 8.0 / (capacity * window_s) } else { f64::NAN },
            rate(received),
            rate(arrived),
            rate(own_bytes),
            flows.len() as f64,
            ratio(dropped(true), own),
            ratio(random, own + random),
            if offered > 0.0 { cross as f64 / offered } else { f64::NAN },
        ]
    }

    /// Opportunities at trace ms `m`, the trace repeating with period P = its
    /// last index (so the last ms of a cycle and ms 0 of the next coincide).
    fn opportunities_at(&self, m: u64) -> u64 {
        let period = self.path.trace.len().saturating_sub(1) as u64;
        if period == 0 {
            return 0;
        }
        let at = |i: u64| {
            usize::try_from(i).ok().and_then(|i| self.path.trace.get(i)).copied().map_or(0, u64::from)
        };
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
