//! Cross traffic: competing flows from the sender namespace to the receiver
//! namespace through the same bottleneck as the observed flow. Flow i uses
//! sender port CROSS_PORT_BASE + i; the observed flow keeps
//! traffic::SENDER_ADDR's port, so link records tell the flows apart.
//!
//! The main thread creates every socket inside its namespace; threads then
//! drive them on absolute deadlines (virtual time under UML), never spinning.
//! `Running::finish` shuts every socket down, so no thread can stay blocked.
//!
//! A TCP flow sends only inside its planned window. Its receiver connects at
//! the start, so the flow really begins when the handshake completes, which
//! a standing queue can delay; that time is reported. At the stop both ends
//! are aborted (sys::abort): the sender's backlog is discarded, not sent,
//! and a connection that completes later is refused.

use crate::netns::Netns;
use crate::plan::CrossFlow;
use crate::step;
use crate::sys::{self, Context, Result, fail};
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
use std::os::fd::{AsFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::JoinHandle;

pub const CROSS_PORT_BASE: u16 = 6000;
const SND: Ipv4Addr = Ipv4Addr::new(10, 0, 0, 1);
const RCV: Ipv4Addr = Ipv4Addr::new(10, 0, 1, 1);
/// UDP payload of a 1500-byte IPv4 packet.
const UDP_PAYLOAD: usize = 1472;
const CHUNK: usize = 256 << 10;

/// Sockets to shut down when the episode ends.
enum Handle {
    Tcp { listener: TcpListener, client: Arc<OwnedFd>, accepted: Arc<OnceLock<Arc<TcpStream>>> },
    Udp { receiver: Arc<UdpSocket> },
}

/// A TCP flow's threads: the sender (bytes written and when it accepted the
/// connection, if it did before the stop), the receiver, the stopper and,
/// for an agent flow, its step loop; and the controller its listener holds.
struct TcpThreads {
    cc: String,
    send: JoinHandle<Result<(u64, Option<u64>)>>,
    recv: JoinHandle<Result<u64>>,
    stop: JoinHandle<Result<u64>>,
    steps: Option<JoinHandle<Result<step::Steps>>>,
}

enum Threads {
    Tcp(TcpThreads),
    Udp { tx: JoinHandle<Result<u64>>, rx: JoinHandle<Result<u64>> },
}

pub struct Running {
    handles: Vec<Handle>,
    threads: Vec<Threads>,
}

/// What one flow did: bytes its sender wrote, the congestion control its
/// socket ran ("" for UDP), when its sender accepted the connection
/// (CLOCK_MONOTONIC ns; None for UDP, or a TCP flow whose handshake did not
/// complete before its stop), and its steps and actions (0 unless a policy
/// drove it).
pub struct FlowReport {
    pub sent: u64,
    pub cc: String,
    pub connected_ns: Option<u64>,
    pub steps: u64,
    pub actions: u64,
}

fn spawn<T: Send + 'static>(
    name: String,
    f: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<JoinHandle<Result<T>>> {
    std::thread::Builder::new().name(name.clone()).spawn(f).ctx(format!("spawning {name}"))
}

fn join<T>(h: JoinHandle<Result<T>>, what: &str) -> Result<T> {
    match h.join() {
        Ok(r) => r.ctx(what.to_owned()),
        Err(_) => fail(format!("{what} thread panicked")),
    }
}

/// The sender port of flow `i`.
pub fn port(i: usize) -> Result<u16> {
    match u16::try_from(i).ok().and_then(|i| CROSS_PORT_BASE.checked_add(i)) {
        Some(p) => Ok(p),
        None => fail(format!("cross flow index {i}")),
    }
}

/// Create every flow's sockets and start its threads; `loops[i]` is flow
/// i's step loop when a policy drives it. Times are trace ms.
pub fn start(
    flows: &[CrossFlow],
    mut loops: Vec<Option<step::Loop>>,
    base_ns: u64,
    snd: &Netns,
    rcv: &Netns,
    home: &Netns,
) -> Result<Running> {
    if loops.len() != flows.len() {
        return fail("one step loop slot per cross flow");
    }
    let mut running = Running { handles: Vec::new(), threads: Vec::new() };
    let at = |ms: u32| base_ns + u64::from(ms) * 1_000_000;
    for (i, flow) in flows.iter().enumerate() {
        let port = port(i)?;
        match flow {
            CrossFlow::Tcp { cc, start_ms, stop_ms, .. } => {
                let listener = snd.within(home, || {
                    let l = TcpListener::bind(SocketAddrV4::new(SND, port)).ctx("binding cross sender")?;
                    sys::setsockopt_bytes(l.as_fd(), libc::IPPROTO_TCP, libc::TCP_CONGESTION, cc.as_bytes())
                        .ctx(format!("cross TCP_CONGESTION {cc}"))?;
                    Ok(l)
                })?;
                // An accepted socket inherits the listener's controller, so
                // a flow that never connects still has a verified one.
                let listener_cc = sys::congestion_control(listener.as_fd())?;
                let stopped = Arc::new(AtomicBool::new(false));
                let client = Arc::new(rcv.within(home, || sys::tcp_socket().ctx("cross receiver socket"))?);
                let accepted: Arc<OnceLock<Arc<TcpStream>>> = Arc::new(OnceLock::new());
                // Step loop: starts once the sender has accepted.
                let (to_steps, from_sender) = mpsc::channel::<Arc<TcpStream>>();
                let steps = match loops.get_mut(i).and_then(Option::take) {
                    Some(lp) => Some(spawn(format!("cross{i}-steps"), move || match from_sender.recv() {
                        Ok(stream) => lp.run(&stream),
                        Err(_) => Ok(step::Steps { count: 0, actions: 0 }),
                    })?),
                    None => None,
                };
                // Sender: accept, then write until shut down.
                let l = listener.try_clone().ctx("cloning listener")?;
                let acc = Arc::clone(&accepted);
                let (halt, want) = (Arc::clone(&stopped), listener_cc.clone());
                let send = spawn(format!("cross{i}-send"), move || {
                    let Ok((stream, _)) = l.accept() else { return Ok((0, None)) };
                    let connected_ns = sys::monotonic_ns();
                    let cc = sys::congestion_control(stream.as_fd())?;
                    if cc != want {
                        return fail(format!("cross flow {i} runs {cc:?}, its listener {want:?}"));
                    }
                    let stream = Arc::clone(acc.get_or_init(|| Arc::new(stream)));
                    // The stopper sets the flag before it looks for this
                    // socket, so one of the two always aborts a connection
                    // that completes around the stop.
                    if halt.load(Ordering::SeqCst) {
                        sys::abort(stream.as_fd());
                        return Ok((0, None));
                    }
                    // A flow without a step loop has no receiver: ignore.
                    let _ = to_steps.send(Arc::clone(&stream));
                    let chunk = vec![0x33u8; CHUNK];
                    let mut sent = 0u64;
                    loop {
                        match (&*stream).write(&chunk) {
                            Ok(n) => sent += n as u64,
                            Err(e) if e.kind() == ErrorKind::Interrupted => {}
                            Err(_) => return Ok((sent, Some(connected_ns))),
                        }
                    }
                })?;
                // Receiver: connect at start, read until the stream ends.
                let fd = Arc::clone(&client);
                let start_ns = at(*start_ms);
                let recv = spawn(format!("cross{i}-recv"), move || {
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
                })?;
                // Stopper: end the flow at stop_ms, whatever state it is in.
                let acc = Arc::clone(&accepted);
                let l = listener.try_clone().ctx("cloning listener")?;
                let fd = Arc::clone(&client);
                let stop_ns = at(*stop_ms);
                let stop = spawn(format!("cross{i}-stop"), move || {
                    sys::sleep_until(stop_ns);
                    stopped.store(true, Ordering::SeqCst);
                    sys::shutdown(l.as_fd()); // no handshake completes from here on
                    if let Some(s) = acc.get() {
                        sys::abort(s.as_fd());
                    }
                    sys::abort(fd.as_fd());
                    Ok(0)
                })?;
                running.handles.push(Handle::Tcp { listener, client, accepted });
                running.threads.push(Threads::Tcp(TcpThreads { cc: listener_cc, send, recv, stop, steps }));
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
                let r = Arc::clone(&receiver);
                let rx = spawn(format!("cross{i}-udprx"), move || {
                    let mut buf = vec![0u8; 2048];
                    while matches!(r.recv(&mut buf), Ok(n) if n > 0) {}
                    Ok(0)
                })?;
                // One 1500-byte packet every interval, from start to stop.
                let interval_ns = u64::try_from(1500u128 * 8 * 1_000_000_000 / u128::from(rate_bps.get()))
                    .ctx("cross interval")?
                    .max(1);
                let (start_ns, stop_ns) = (at(*start_ms), at(*stop_ms));
                let tx = spawn(format!("cross{i}-udptx"), move || {
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
                })?;
                running.handles.push(Handle::Udp { receiver });
                running.threads.push(Threads::Udp { tx, rx });
            }
        }
    }
    Ok(running)
}

impl Running {
    /// Shut every flow down and report each, in flow order.
    pub fn finish(self) -> Result<Vec<FlowReport>> {
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
        self.threads
            .into_iter()
            .map(|t| match t {
                Threads::Tcp(t) => {
                    let (sent, connected_ns) = join(t.send, "cross sender")?;
                    join(t.recv, "cross receiver")?;
                    join(t.stop, "cross stopper")?;
                    let s = match t.steps {
                        Some(h) => join(h, "cross step loop")?,
                        None => step::Steps { count: 0, actions: 0 },
                    };
                    Ok(FlowReport { sent, cc: t.cc, connected_ns, steps: s.count, actions: s.actions })
                }
                Threads::Udp { tx, rx } => {
                    let sent = join(tx, "cross udp sender")?;
                    join(rx, "cross udp receiver")?;
                    Ok(FlowReport { sent, cc: String::new(), connected_ns: None, steps: 0, actions: 0 })
                }
            })
            .collect()
    }
}
