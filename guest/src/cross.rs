//! Cross traffic: competing flows from the sender namespace to the receiver
//! namespace through the same bottleneck as the observed flow. Flow i uses
//! sender port CROSS_PORT_BASE + i; the observed flow keeps
//! traffic::SENDER_ADDR's port, so link records tell the flows apart.
//!
//! The main thread creates every socket inside its namespace; threads then
//! drive them on absolute deadlines (virtual time under UML), never spinning.
//! `Running::finish` shuts every socket down, so no thread can stay blocked.

use crate::netns::Netns;
use crate::plan::CrossFlow;
use crate::sys::{self, Context, Result, fail};
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
use std::os::fd::{AsFd, OwnedFd};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

pub const CROSS_PORT_BASE: u16 = 6000;
const SND: Ipv4Addr = Ipv4Addr::new(10, 0, 0, 1);
const RCV: Ipv4Addr = Ipv4Addr::new(10, 0, 1, 1);
/// UDP payload of a 1500-byte IPv4 packet.
const UDP_PAYLOAD: usize = 1472;
const CHUNK: usize = 256 << 10;

/// Sockets to shut down when the episode ends.
enum Handle {
    Tcp { listener: TcpListener, client: Arc<OwnedFd>, accepted: Arc<OnceLock<TcpStream>> },
    Udp { receiver: Arc<UdpSocket> },
}

pub struct Running {
    handles: Vec<Handle>,
    threads: Vec<JoinHandle<Result<u64>>>,
}

fn spawn(name: String, f: impl FnOnce() -> Result<u64> + Send + 'static) -> Result<JoinHandle<Result<u64>>> {
    std::thread::Builder::new().name(name.clone()).spawn(f).ctx(format!("spawning {name}"))
}

/// Create every flow's sockets and start its threads. Times are trace ms.
pub fn start(flows: &[CrossFlow], base_ns: u64, snd: &Netns, rcv: &Netns, home: &Netns) -> Result<Running> {
    let mut running = Running { handles: Vec::new(), threads: Vec::new() };
    let at = |ms: u32| base_ns + u64::from(ms) * 1_000_000;
    for (i, flow) in flows.iter().enumerate() {
        let port = CROSS_PORT_BASE + u16::try_from(i).ctx("cross flow index")?;
        match flow {
            CrossFlow::Tcp { cc, start_ms, stop_ms } => {
                let listener = snd.within(home, || {
                    let l = TcpListener::bind(SocketAddrV4::new(SND, port)).ctx("binding cross sender")?;
                    sys::setsockopt_bytes(l.as_fd(), libc::IPPROTO_TCP, libc::TCP_CONGESTION, cc.as_bytes())
                        .ctx(format!("cross TCP_CONGESTION {cc}"))?;
                    Ok(l)
                })?;
                let client = Arc::new(rcv.within(home, || sys::tcp_socket().ctx("cross receiver socket"))?);
                let accepted = Arc::new(OnceLock::new());
                // Sender: accept, then write until shut down.
                let l = listener.try_clone().ctx("cloning listener")?;
                let acc = Arc::clone(&accepted);
                running.threads.push(spawn(format!("cross{i}-send"), move || {
                    let Ok((stream, _)) = l.accept() else { return Ok(0) };
                    let stream = acc.get_or_init(|| stream);
                    let chunk = vec![0x33u8; CHUNK];
                    let mut sent = 0u64;
                    loop {
                        match (&*stream).write(&chunk) {
                            Ok(n) => sent += n as u64,
                            Err(e) if e.kind() == ErrorKind::Interrupted => {}
                            Err(_) => return Ok(sent),
                        }
                    }
                })?);
                // Receiver: connect at start, read until the stream ends.
                let fd = Arc::clone(&client);
                let start_ns = at(*start_ms);
                running.threads.push(spawn(format!("cross{i}-recv"), move || {
                    sys::sleep_until(start_ns);
                    if sys::connect_v4(fd.as_fd(), SocketAddrV4::new(SND, port)).is_err() {
                        return Ok(0); // shut down before it could connect
                    }
                    let mut buf = vec![0u8; CHUNK];
                    let stream = TcpStream::from(fd.as_fd().try_clone_to_owned().ctx("dup")?);
                    loop {
                        match (&stream).read(&mut buf) {
                            Ok(0) | Err(_) => return Ok(0),
                            Ok(_) => {}
                        }
                    }
                })?);
                // Stopper: end the flow at stop_ms.
                let acc = Arc::clone(&accepted);
                let stop_ns = at(*stop_ms);
                running.threads.push(spawn(format!("cross{i}-stop"), move || {
                    sys::sleep_until(stop_ns);
                    if let Some(s) = acc.get() {
                        sys::shutdown(s.as_fd());
                    }
                    Ok(0)
                })?);
                running.handles.push(Handle::Tcp { listener, client, accepted });
            }
            CrossFlow::Udp { rate_bps, start_ms, stop_ms } => {
                let receiver =
                    Arc::new(rcv.within(home, || {
                        UdpSocket::bind(SocketAddrV4::new(RCV, port)).ctx("cross udp rx")
                    })?);
                let sender = snd.within(home, || {
                    let s = UdpSocket::bind(SocketAddrV4::new(SND, port)).ctx("cross udp tx")?;
                    s.connect(SocketAddrV4::new(RCV, port)).ctx("cross udp connect")?;
                    Ok(s)
                })?;
                let rx = Arc::clone(&receiver);
                running.threads.push(spawn(format!("cross{i}-udprx"), move || {
                    let mut buf = vec![0u8; 2048];
                    while matches!(rx.recv(&mut buf), Ok(n) if n > 0) {}
                    Ok(0)
                })?);
                // One 1500-byte packet every interval, from start to stop.
                let interval_ns = u64::try_from(1500u128 * 8 * 1_000_000_000 / u128::from(rate_bps.get()))
                    .ctx("cross interval")?
                    .max(1);
                let (start_ns, stop_ns) = (at(*start_ms), at(*stop_ms));
                running.threads.push(spawn(format!("cross{i}-udptx"), move || {
                    let payload = [0x44u8; UDP_PAYLOAD];
                    let mut sent = 0u64;
                    let mut t = start_ns;
                    while t < stop_ns {
                        sys::sleep_until(t);
                        match sender.send(&payload) {
                            Ok(n) => sent += n as u64,
                            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                            // A full socket buffer drops the packet, as a
                            // CBR source would; anything else is an error.
                            Err(e) if e.raw_os_error() == Some(libc::ENOBUFS) => {}
                            Err(e) => return Err(e).ctx("cross udp send"),
                        }
                        t += interval_ns;
                    }
                    Ok(sent)
                })?);
                running.handles.push(Handle::Udp { receiver });
            }
        }
    }
    Ok(running)
}

impl Running {
    /// Shut every flow down and return the bytes each sender wrote, in
    /// flow order (TCP flows run three threads, UDP flows two).
    pub fn finish(self) -> Result<Vec<u64>> {
        for h in &self.handles {
            match h {
                Handle::Tcp { listener, client, accepted } => {
                    sys::shutdown(listener.as_fd());
                    sys::shutdown(client.as_fd());
                    if let Some(s) = accepted.get() {
                        sys::shutdown(s.as_fd());
                    }
                }
                Handle::Udp { receiver } => sys::shutdown(receiver.as_fd()),
            }
        }
        let mut totals = Vec::with_capacity(self.handles.len());
        let mut threads = self.threads.into_iter();
        for h in &self.handles {
            let per_flow = match h {
                Handle::Tcp { .. } => 3,
                Handle::Udp { .. } => 2,
            };
            let mut sent = 0;
            for _ in 0..per_flow {
                let Some(t) = threads.next() else { return fail("cross thread missing") };
                sent += match t.join() {
                    Ok(r) => r?,
                    Err(_) => return fail("cross thread panicked"),
                };
            }
            totals.push(sent);
        }
        Ok(totals)
    }
}
