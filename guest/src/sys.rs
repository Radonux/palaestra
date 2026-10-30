//! Thin safe wrappers over the raw syscalls the runner needs, plus the error
//! type. Every `unsafe` block is a single libc call whose pointer arguments
//! come from live Rust values or NUL-terminated `CString`s.

use std::ffi::CString;
use std::fmt;
use std::io;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::path::Path;
use std::time::Duration;

/// An error with the chain of things the runner was doing when it happened.
#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Context<T> {
    fn ctx(self, what: impl fmt::Display) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn ctx(self, what: impl fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{what}: {e}")))
    }
}

pub fn fail<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

fn check(ret: libc::c_int) -> io::Result<libc::c_int> {
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(ret) }
}

fn cstr(s: &str) -> io::Result<CString> {
    CString::new(s).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
}

fn cpath(p: &Path) -> io::Result<CString> {
    cstr(&p.to_string_lossy())
}

/// mount(2). `EBUSY` (already mounted, e.g. devtmpfs automount) is success.
pub fn mount(source: &str, target: &str, fstype: &str, data: Option<&str>) -> io::Result<()> {
    let (s, t, f) = (cstr(source)?, cstr(target)?, cstr(fstype)?);
    let d = data.map(cstr).transpose()?;
    let dptr = d.as_ref().map_or(std::ptr::null(), |d| d.as_ptr().cast());
    // SAFETY: all pointers are NUL-terminated strings alive for the call.
    let ret = unsafe { libc::mount(s.as_ptr(), t.as_ptr(), f.as_ptr(), 0, dptr) };
    match check(ret) {
        Err(e) if e.raw_os_error() == Some(libc::EBUSY) => Ok(()),
        r => r.map(drop),
    }
}

/// finit_module(2) with a parameter string. `EEXIST` is reported to the caller.
pub fn finit_module(path: &Path, params: &str) -> io::Result<()> {
    let file = std::fs::File::open(path)?;
    let p = cstr(params)?;
    // SAFETY: fd is open for the call; params is NUL-terminated.
    let ret = unsafe { libc::syscall(libc::SYS_finit_module, file.as_raw_fd(), p.as_ptr(), 0) };
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// unshare(CLONE_NEWNET) for the calling thread.
pub fn unshare_net() -> io::Result<()> {
    // SAFETY: no pointer arguments.
    check(unsafe { libc::unshare(libc::CLONE_NEWNET) }).map(drop)
}

/// setns(fd, CLONE_NEWNET) for the calling thread.
pub fn setns_net(fd: BorrowedFd<'_>) -> io::Result<()> {
    // SAFETY: fd is a valid open descriptor borrowed for the call.
    check(unsafe { libc::setns(fd.as_raw_fd(), libc::CLONE_NEWNET) }).map(drop)
}

pub fn open_path(path: &Path, flags: libc::c_int) -> io::Result<OwnedFd> {
    let p = cpath(path)?;
    // SAFETY: p is NUL-terminated; a non-negative return is a fresh fd we own.
    let fd = check(unsafe { libc::open(p.as_ptr(), flags | libc::O_CLOEXEC) })?;
    // SAFETY: fd was just returned by open and is owned by nobody else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: ts is a valid out-pointer; CLOCK_MONOTONIC cannot fail.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    // Monotonic time is non-negative, so the casts are lossless.
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

/// Sleep until CLOCK_MONOTONIC reaches `deadline_ns` (virtual time in UML).
pub fn sleep_until(deadline_ns: u64) {
    let ts = libc::timespec {
        tv_sec: (deadline_ns / 1_000_000_000) as _,
        tv_nsec: (deadline_ns % 1_000_000_000) as libc::c_long,
    };
    loop {
        // SAFETY: ts is valid; the remainder pointer may be null for TIMER_ABSTIME.
        let ret = unsafe {
            libc::clock_nanosleep(libc::CLOCK_MONOTONIC, libc::TIMER_ABSTIME, &ts, std::ptr::null_mut())
        };
        if ret != libc::EINTR {
            return;
        }
    }
}

pub fn sleep(d: Duration) {
    sleep_until(monotonic_ns() + u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
}

pub fn setsockopt_bytes(fd: BorrowedFd<'_>, level: i32, name: i32, val: &[u8]) -> io::Result<()> {
    let len =
        libc::socklen_t::try_from(val.len()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: val points to len readable bytes for the call.
    check(unsafe { libc::setsockopt(fd.as_raw_fd(), level, name, val.as_ptr().cast(), len) }).map(drop)
}

pub fn setsockopt_int(fd: BorrowedFd<'_>, level: i32, name: i32, val: libc::c_int) -> io::Result<()> {
    setsockopt_bytes(fd, level, name, &val.to_ne_bytes())
}

pub fn getsockopt_bytes(fd: BorrowedFd<'_>, level: i32, name: i32, buf: &mut [u8]) -> io::Result<usize> {
    let mut len =
        libc::socklen_t::try_from(buf.len()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    // SAFETY: buf has len writable bytes; len is a valid in/out pointer.
    check(unsafe { libc::getsockopt(fd.as_raw_fd(), level, name, buf.as_mut_ptr().cast(), &mut len) })?;
    Ok(len as usize)
}

pub fn shutdown(fd: BorrowedFd<'_>) {
    // SAFETY: no pointer arguments. Errors (already closed) are irrelevant.
    unsafe { libc::shutdown(fd.as_raw_fd(), libc::SHUT_RDWR) };
}

/// Flush filesystems and power the UML guest off. Only valid as PID 1.
pub fn power_off() -> ! {
    // SAFETY: no pointer arguments.
    unsafe {
        libc::sync();
        libc::reboot(libc::LINUX_REBOOT_CMD_POWER_OFF);
    }
    // reboot only returns on failure; PID 1 must never exit, so park.
    loop {
        sleep(Duration::from_secs(3600));
    }
}

pub fn uname() -> (String, String) {
    // SAFETY: utsname is plain old data; zeroed is a valid value.
    let mut u: libc::utsname = unsafe { std::mem::zeroed() };
    // SAFETY: u is a valid out-pointer.
    if unsafe { libc::uname(&mut u) } != 0 {
        return (String::new(), String::new());
    }
    let field = |f: &[libc::c_char]| {
        let bytes: Vec<u8> = f.iter().take_while(|&&c| c != 0).map(|&c| c.to_ne_bytes()[0]).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    (field(&u.release), field(&u.version))
}

/// An unconnected IPv4 TCP socket in the calling thread's namespace.
pub fn tcp_socket() -> io::Result<OwnedFd> {
    // SAFETY: plain socket(2); a non-negative return is a new fd we own.
    let fd = check(unsafe { libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) })?;
    // SAFETY: fd was just returned by socket and is owned by nobody else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// connect(2) to an IPv4 address; blocks until connected or failed.
pub fn connect_v4(fd: BorrowedFd<'_>, addr: std::net::SocketAddrV4) -> io::Result<()> {
    let sa = libc::sockaddr_in {
        sin_family: libc::AF_INET as libc::sa_family_t,
        sin_port: addr.port().to_be(),
        sin_addr: libc::in_addr { s_addr: u32::from_ne_bytes(addr.ip().octets()) },
        sin_zero: [0; 8],
    };
    let len = std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t;
    loop {
        // SAFETY: sa is a valid sockaddr_in of len bytes for the call.
        let ret = unsafe { libc::connect(fd.as_raw_fd(), (&raw const sa).cast(), len) };
        match check(ret) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            r => return r.map(drop),
        }
    }
}
