//! The bulk TCP pair, mirroring Mahak's harness: the sender listens with the
//! congestion control and observation mark set on the listening socket (so
//! the accepted socket inherits both from the handshake on), the receiver
//! connects at trace time 0, and the sender writes until it is shut down.
//! Both sides set TCP_NODELAY as Mahak's server and client do.

use crate::sys::{self, Context, Result, fail};
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::os::fd::AsFd;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;

pub const SENDER_ADDR: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 5001);
const CHUNK: usize = 256 << 10;

/// What the sender verified about its accepted socket.
pub struct SenderReport {
    pub bytes: u64,
    pub cc: String,
}

pub struct Sender {
    pub handle: JoinHandle<Result<SenderReport>>,
    /// The accepted connection, available once the handshake completes.
    pub conn: mpsc::Receiver<Arc<TcpStream>>,
}

/// Bind and listen now, in the caller's namespace; accept and send on a
/// thread (the accepted socket stays in the listener's namespace).
pub fn start_sender(cc: &str, mark: u32) -> Result<Sender> {
    let listener = TcpListener::bind(SENDER_ADDR).ctx("binding sender")?;
    let fd = listener.as_fd();
    sys::setsockopt_bytes(fd, libc::IPPROTO_TCP, libc::TCP_CONGESTION, cc.as_bytes())
        .ctx(format!("TCP_CONGESTION {cc}"))?;
    sys::setsockopt_int(fd, libc::SOL_SOCKET, libc::SO_MARK, mark as libc::c_int).ctx("SO_MARK")?;
    sys::setsockopt_int(fd, libc::IPPROTO_TCP, libc::TCP_NODELAY, 1).ctx("TCP_NODELAY")?;
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("sender".into())
        .spawn(move || -> Result<SenderReport> {
            let (stream, _) = listener.accept().ctx("accept")?;
            let mut name = [0u8; 16];
            let n = sys::getsockopt_bytes(stream.as_fd(), libc::IPPROTO_TCP, libc::TCP_CONGESTION, &mut name)
                .ctx("reading TCP_CONGESTION")?;
            let cc =
                String::from_utf8_lossy(name.get(..n).unwrap_or_default()).trim_end_matches('\0').to_owned();
            let stream = Arc::new(stream);
            tx.send(Arc::clone(&stream)).ctx("publishing connection")?;
            let chunk = vec![0x5au8; CHUNK];
            let mut bytes = 0u64;
            loop {
                match (&*stream).write(&chunk) {
                    Ok(n) => bytes += n as u64,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    // Shut down at episode end: the expected way out.
                    Err(_) => break,
                }
            }
            Ok(SenderReport { bytes, cc })
        })
        .ctx("spawning sender")?;
    Ok(Sender { handle, conn: rx })
}

pub struct Receiver {
    pub handle: JoinHandle<Result<u64>>,
    pub conn: Arc<TcpStream>,
}

/// Connect now, from the caller's namespace, and read on a thread until the
/// stream ends.
pub fn start_receiver() -> Result<Receiver> {
    let stream = TcpStream::connect(SENDER_ADDR).ctx("connecting receiver")?;
    stream.set_nodelay(true).ctx("TCP_NODELAY")?;
    let stream = Arc::new(stream);
    let conn = Arc::clone(&stream);
    let handle = std::thread::Builder::new()
        .name("receiver".into())
        .spawn(move || -> Result<u64> {
            let mut buf = vec![0u8; CHUNK];
            let mut bytes = 0u64;
            loop {
                match (&*stream).read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => bytes += n as u64,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) if e.kind() == ErrorKind::ConnectionReset => break,
                    Err(e) => return Err(e).ctx("receiving"),
                }
            }
            Ok(bytes)
        })
        .ctx("spawning receiver")?;
    Ok(Receiver { handle, conn })
}

pub fn join<T>(h: JoinHandle<Result<T>>, what: &str) -> Result<T> {
    match h.join() {
        Ok(r) => r.ctx(what),
        Err(_) => fail(format!("{what} thread panicked")),
    }
}
