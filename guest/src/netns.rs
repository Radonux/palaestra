//! Network namespaces held as file descriptors. The runner's main thread
//! creates them and switches between them; worker threads enter theirs
//! before opening sockets, since a socket stays in the namespace it was
//! created in.

use crate::sys::{self, Context, Result};
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::Path;

pub struct Netns {
    fd: OwnedFd,
}

impl Netns {
    /// The calling thread's current namespace.
    pub fn current() -> Result<Netns> {
        let fd =
            sys::open_path(Path::new("/proc/thread-self/ns/net"), libc::O_RDONLY).ctx("opening netns")?;
        Ok(Netns { fd })
    }

    /// A fresh namespace; the calling thread ends up back in `home`.
    pub fn create(home: &Netns) -> Result<Netns> {
        sys::unshare_net().ctx("unshare(CLONE_NEWNET)")?;
        let ns = Netns::current();
        home.enter()?;
        ns
    }

    pub fn enter(&self) -> Result<()> {
        sys::setns_net(self.fd.as_fd()).ctx("setns")
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Run `f` inside this namespace, returning to `home` even if `f` fails.
    pub fn within<T>(&self, home: &Netns, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.enter()?;
        let r = f();
        home.enter()?;
        r
    }
}
