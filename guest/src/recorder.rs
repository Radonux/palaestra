//! Drains /dev/expl_rec into one file per record type, so the host can map
//! each file as a homogeneous array. Framing is checked here (known type,
//! exact length, contiguous sequence numbers); the host checks it again
//! because it does not trust guest output.
//!
//! The ring's read() never blocks, so this thread sleeps between drains:
//! spinning would distort UML virtual time (docs/m0-report.md, F2).

use crate::sys::{self, Context, Result, fail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, ErrorKind, Read, Write};
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

const POLL: Duration = Duration::from_millis(20);

/// (type, exact length, output file stem); mirrors kernel/modules/expliccit_rec.h.
const TYPES: [(u16, usize, &str); 8] = [
    (1, 32, "meta"),
    (2, 64, "link_arrival"),
    (3, 64, "link_drop"),
    (4, 64, "link_depart"),
    (5, 40, "link_opp"),
    (6, 96, "link_stats"),
    (7, 160, "tcp_obs"),
    (8, 32, "agent_action"),
];

#[derive(Default, Debug)]
pub struct Summary {
    pub counts: BTreeMap<&'static str, u64>,
    pub bytes: u64,
}

pub struct Recorder {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<Result<Summary>>,
}

impl Recorder {
    /// Open the device now (the ring starts empty at open) and drain it on a
    /// thread until `finish`.
    pub fn start(out: &Path) -> Result<Recorder> {
        let fd = sys::open_path(Path::new("/dev/expl_rec"), libc::O_RDONLY).ctx("opening /dev/expl_rec")?;
        let mut writers = Vec::with_capacity(TYPES.len());
        for (_, _, stem) in TYPES {
            let path: PathBuf = out.join(format!("{stem}.bin"));
            writers.push(BufWriter::with_capacity(1 << 20, File::create(&path).ctx(path.display())?));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("recorder".into())
            .spawn(move || drain(fd, writers, &flag))
            .ctx("spawning recorder")?;
        Ok(Recorder { stop, handle })
    }

    /// Stop after one final drain. Call only once every producer is done.
    pub fn finish(self) -> Result<Summary> {
        self.stop.store(true, Ordering::Release);
        match self.handle.join() {
            Ok(r) => r,
            Err(_) => fail("recorder thread panicked"),
        }
    }
}

fn drain(fd: OwnedFd, mut writers: Vec<BufWriter<File>>, stop: &AtomicBool) -> Result<Summary> {
    let mut dev = File::from(fd);
    let mut buf = vec![0u8; 4 << 20];
    let mut pending: Vec<u8> = Vec::with_capacity(1 << 20);
    let mut summary = Summary::default();
    let mut next_seq: Option<u32> = None;
    loop {
        // Read the flag before draining, so records emitted before stop was
        // set are always picked up by this or the next drain.
        let stopping = stop.load(Ordering::Acquire);
        loop {
            match dev.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    pending.extend_from_slice(buf.get(..n).unwrap_or_default());
                    summary.bytes += n as u64;
                    let used = split(&pending, &mut writers, &mut summary, &mut next_seq)?;
                    pending.drain(..used);
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e).ctx("reading /dev/expl_rec"),
            }
        }
        if stopping {
            break;
        }
        sys::sleep(POLL);
    }
    if !pending.is_empty() {
        return fail(format!("{} trailing bytes do not form a record", pending.len()));
    }
    for w in &mut writers {
        w.flush().ctx("flushing records")?;
    }
    Ok(summary)
}

/// Write every complete record in `b` to its file; return bytes consumed.
fn split(
    b: &[u8],
    writers: &mut [BufWriter<File>],
    summary: &mut Summary,
    next_seq: &mut Option<u32>,
) -> Result<usize> {
    let mut at = 0;
    while let Some(hdr) = b.get(at..at + 8).and_then(|h| <[u8; 8]>::try_from(h).ok()) {
        let ty = u16::from_le_bytes([hdr[0], hdr[1]]);
        let len = usize::from(u16::from_le_bytes([hdr[2], hdr[3]]));
        let seq = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        let Some(i) = TYPES.iter().position(|&(t, _, _)| t == ty) else {
            return fail(format!("unknown record type {ty} at stream offset {}", summary.bytes));
        };
        let (_, want, stem) = TYPES.get(i).copied().unwrap_or((0, 0, ""));
        if len != want {
            return fail(format!("record type {ty} has length {len}, expected {want}"));
        }
        let Some(rec) = b.get(at..at + len) else { break };
        if let Some(expected) = *next_seq
            && seq != expected
        {
            return fail(format!("record sequence jumped from {expected} to {seq}: ring overflowed"));
        }
        *next_seq = Some(seq.wrapping_add(1));
        let Some(w) = writers.get_mut(i) else { return fail("writer missing") };
        w.write_all(rec).ctx("writing record")?;
        *summary.counts.entry(stem).or_default() += 1;
        at += len;
    }
    Ok(at)
}
