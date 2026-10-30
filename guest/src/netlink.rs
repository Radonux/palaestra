//! Minimal rtnetlink client: exactly the requests the runner makes (veth,
//! link up, IPv4 address, default route, mmlink qdisc add and delete).
//! Constants are copied from the uapi headers named next to them, so the
//! runner does not depend on how complete libc's tables are per target.
//! The mmlink attributes mirror kernel/modules/sch_mmlink.h.

use crate::sys::{Context, Error, Result, fail};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

// include/uapi/linux/netlink.h
const NETLINK_ROUTE: i32 = 0;
const NLMSG_ERROR: u16 = 2;
const NLM_F_REQUEST: u16 = 0x1;
const NLM_F_ACK: u16 = 0x4;
const NLM_F_EXCL: u16 = 0x200;
const NLM_F_CREATE: u16 = 0x400;
const NLM_F_ACK_TLVS: u16 = 0x200;
const NLA_F_NESTED: u16 = 0x8000;
const NLMSGERR_ATTR_MSG: u16 = 1;
const SOL_NETLINK: i32 = 270;
const NETLINK_CAP_ACK: i32 = 10;
const NETLINK_EXT_ACK: i32 = 11;
const NLMSG_HDRLEN: usize = 16;
// include/uapi/linux/rtnetlink.h
const RTM_NEWLINK: u16 = 16;
const RTM_NEWADDR: u16 = 20;
const RTM_NEWROUTE: u16 = 24;
const RTM_NEWQDISC: u16 = 36;
const RTM_DELQDISC: u16 = 37;
const RTA_GATEWAY: u16 = 5;
const RT_TABLE_MAIN: u8 = 254;
const RTPROT_BOOT: u8 = 3;
const RT_SCOPE_UNIVERSE: u8 = 0;
const RTN_UNICAST: u8 = 1;
const TCA_KIND: u16 = 1;
const TCA_OPTIONS: u16 = 2;
// include/uapi/linux/if_link.h, if_addr.h, veth.h, pkt_sched.h
const IFLA_IFNAME: u16 = 3;
const IFLA_LINKINFO: u16 = 18;
const IFLA_NET_NS_FD: u16 = 28;
const IFLA_INFO_KIND: u16 = 1;
const IFLA_INFO_DATA: u16 = 2;
const VETH_INFO_PEER: u16 = 1;
const IFA_ADDRESS: u16 = 1;
const IFA_LOCAL: u16 = 2;
const TC_H_ROOT: u32 = 0xFFFF_FFFF;
const AF_UNSPEC: u8 = 0;
const AF_INET: u8 = 2;
const IFF_UP: u32 = 1;
// kernel/modules/sch_mmlink.h
const TCA_MMLINK_BASE_NS: u16 = 1;
const TCA_MMLINK_DELAY_NS: u16 = 2;
const TCA_MMLINK_ORDER: u16 = 3;
const TCA_MMLINK_TRACE: u16 = 4;
const TCA_MMLINK_LIMIT_PKTS: u16 = 5;
const TCA_MMLINK_LIMIT_BYTES: u16 = 6;
const TCA_MMLINK_LINK_ID: u16 = 8;
const TCA_MMLINK_LOSS: u16 = 9;

/// Stage order of one mmlink qdisc (enum mmlink_order).
#[derive(Clone, Copy)]
pub enum Order {
    DelayLink = 0,
    LinkDelay = 1,
}

pub struct Mmlink<'a> {
    pub link_id: u32,
    pub order: Order,
    pub base_ns: u64,
    pub delay_ns: u64,
    pub trace: &'a [u8],
    pub limit_pkts: u32,
    pub limit_bytes: u32,
    pub loss: u32,
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// A request under construction. Lengths are patched when nests close and
/// when the socket sends it.
struct Msg {
    buf: Vec<u8>,
}

impl Msg {
    fn new(ty: u16, flags: u16, family_hdr: &[u8]) -> Msg {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&0u32.to_ne_bytes());
        buf.extend_from_slice(&ty.to_ne_bytes());
        buf.extend_from_slice(&(flags | NLM_F_REQUEST | NLM_F_ACK).to_ne_bytes());
        buf.extend_from_slice(&[0; 8]); // seq and portid, set on send
        let mut m = Msg { buf };
        m.raw(family_hdr);
        m
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        self.buf.resize(align4(self.buf.len()), 0);
    }

    fn attr(&mut self, ty: u16, data: &[u8]) -> Result<()> {
        let len = u16::try_from(4 + data.len()).ctx("netlink attribute too large")?;
        self.buf.extend_from_slice(&len.to_ne_bytes());
        self.buf.extend_from_slice(&ty.to_ne_bytes());
        self.raw(data);
        Ok(())
    }

    fn nest(&mut self, ty: u16) -> usize {
        let at = self.buf.len();
        self.buf.extend_from_slice(&0u16.to_ne_bytes());
        self.buf.extend_from_slice(&(ty | NLA_F_NESTED).to_ne_bytes());
        at
    }

    fn end(&mut self, at: usize) -> Result<()> {
        let len = u16::try_from(self.buf.len() - at).ctx("netlink nest too large")?;
        let Some(slot) = self.buf.get_mut(at..at + 2) else { return fail("bad nest offset") };
        slot.copy_from_slice(&len.to_ne_bytes());
        Ok(())
    }
}

fn ifinfomsg(index: i32, flags: u32, change: u32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0] = AF_UNSPEC; // 16-byte literal array: indices are in bounds.
    h[4..8].copy_from_slice(&index.to_ne_bytes());
    h[8..12].copy_from_slice(&flags.to_ne_bytes());
    h[12..16].copy_from_slice(&change.to_ne_bytes());
    h
}

fn tcmsg(ifindex: u32) -> [u8; 20] {
    let mut h = [0u8; 20];
    h[4..8].copy_from_slice(&ifindex.to_ne_bytes());
    h[12..16].copy_from_slice(&TC_H_ROOT.to_ne_bytes()); // parent
    h
}

/// An rtnetlink socket bound to the network namespace that was current
/// when it was opened.
pub struct Socket {
    fd: OwnedFd,
    seq: u32,
}

impl Socket {
    pub fn open() -> Result<Socket> {
        // SAFETY: plain socket(2) call; a non-negative return is a new fd we own.
        let raw =
            unsafe { libc::socket(libc::AF_NETLINK, libc::SOCK_RAW | libc::SOCK_CLOEXEC, NETLINK_ROUTE) };
        if raw < 0 {
            return Err(io::Error::last_os_error()).ctx("netlink socket");
        }
        // SAFETY: raw is a fresh descriptor owned by nobody else.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        crate::sys::setsockopt_int(fd.as_fd(), SOL_NETLINK, NETLINK_EXT_ACK, 1).ctx("NETLINK_EXT_ACK")?;
        crate::sys::setsockopt_int(fd.as_fd(), SOL_NETLINK, NETLINK_CAP_ACK, 1).ctx("NETLINK_CAP_ACK")?;
        Ok(Socket { fd, seq: 0 })
    }

    /// Send one request and wait for its ack; a kernel error becomes an Err
    /// carrying errno and the extended-ack message if there is one.
    fn call(&mut self, mut m: Msg) -> Result<()> {
        self.seq += 1;
        let len = u32::try_from(m.buf.len()).ctx("netlink message too large")?;
        if let Some(h) = m.buf.get_mut(0..4) {
            h.copy_from_slice(&len.to_ne_bytes());
        }
        if let Some(h) = m.buf.get_mut(8..12) {
            h.copy_from_slice(&self.seq.to_ne_bytes());
        }
        // SAFETY: buf is valid for len bytes; the kernel is the default peer.
        let sent = unsafe { libc::send(self.fd.as_raw_fd(), m.buf.as_ptr().cast(), m.buf.len(), 0) };
        if sent < 0 {
            return Err(io::Error::last_os_error()).ctx("netlink send");
        }
        let mut buf = vec![0u8; 65536];
        loop {
            // SAFETY: buf is valid for its length.
            let n = unsafe { libc::recv(self.fd.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len(), 0) };
            let n = usize::try_from(n).map_err(|_| io::Error::last_os_error()).ctx("netlink recv")?;
            match parse_ack(buf.get(..n).unwrap_or_default(), self.seq)? {
                Some(r) => return r,
                None => continue,
            }
        }
    }

    pub fn veth(&mut self, a: &str, a_ns: BorrowedFd<'_>, b: &str, b_ns: BorrowedFd<'_>) -> Result<()> {
        let mut m = Msg::new(RTM_NEWLINK, NLM_F_CREATE | NLM_F_EXCL, &ifinfomsg(0, 0, 0));
        m.attr(IFLA_IFNAME, cstr(a).as_bytes())?;
        m.attr(IFLA_NET_NS_FD, &fd_u32(a_ns).to_ne_bytes())?;
        let info = m.nest(IFLA_LINKINFO);
        m.attr(IFLA_INFO_KIND, b"veth\0")?;
        let data = m.nest(IFLA_INFO_DATA);
        let peer = m.nest(VETH_INFO_PEER);
        m.raw(&ifinfomsg(0, 0, 0));
        m.attr(IFLA_IFNAME, cstr(b).as_bytes())?;
        m.attr(IFLA_NET_NS_FD, &fd_u32(b_ns).to_ne_bytes())?;
        m.end(peer)?;
        m.end(data)?;
        m.end(info)?;
        self.call(m).ctx(format!("creating veth {a}/{b}"))
    }

    pub fn link_up(&mut self, name: &str) -> Result<()> {
        let mut m = Msg::new(RTM_NEWLINK, 0, &ifinfomsg(0, IFF_UP, IFF_UP));
        m.attr(IFLA_IFNAME, cstr(name).as_bytes())?;
        self.call(m).ctx(format!("setting {name} up"))
    }

    pub fn add_addr(&mut self, ifindex: u32, addr: [u8; 4], prefix: u8) -> Result<()> {
        let mut h = [0u8; 8];
        h[0] = AF_INET; // 8-byte literal array: indices are in bounds.
        h[1] = prefix;
        h[4..8].copy_from_slice(&ifindex.to_ne_bytes());
        let mut m = Msg::new(RTM_NEWADDR, NLM_F_CREATE | NLM_F_EXCL, &h);
        m.attr(IFA_LOCAL, &addr)?;
        m.attr(IFA_ADDRESS, &addr)?;
        self.call(m).ctx("adding address")
    }

    pub fn add_default_route(&mut self, gateway: [u8; 4]) -> Result<()> {
        let mut h = [0u8; 12];
        h[0] = AF_INET; // 12-byte literal array: indices are in bounds.
        h[4] = RT_TABLE_MAIN;
        h[5] = RTPROT_BOOT;
        h[6] = RT_SCOPE_UNIVERSE;
        h[7] = RTN_UNICAST;
        let mut m = Msg::new(RTM_NEWROUTE, NLM_F_CREATE | NLM_F_EXCL, &h);
        m.attr(RTA_GATEWAY, &gateway)?;
        self.call(m).ctx("adding default route")
    }

    pub fn add_mmlink(&mut self, ifindex: u32, q: &Mmlink<'_>) -> Result<()> {
        let mut m = Msg::new(RTM_NEWQDISC, NLM_F_CREATE | NLM_F_EXCL, &tcmsg(ifindex));
        m.attr(TCA_KIND, b"mmlink\0")?;
        let opts = m.nest(TCA_OPTIONS);
        m.attr(TCA_MMLINK_BASE_NS, &q.base_ns.to_ne_bytes())?;
        m.attr(TCA_MMLINK_DELAY_NS, &q.delay_ns.to_ne_bytes())?;
        m.attr(TCA_MMLINK_ORDER, &[q.order as u8])?;
        m.attr(TCA_MMLINK_LIMIT_PKTS, &q.limit_pkts.to_ne_bytes())?;
        m.attr(TCA_MMLINK_LIMIT_BYTES, &q.limit_bytes.to_ne_bytes())?;
        m.attr(TCA_MMLINK_LINK_ID, &q.link_id.to_ne_bytes())?;
        m.attr(TCA_MMLINK_LOSS, &q.loss.to_ne_bytes())?;
        m.attr(TCA_MMLINK_TRACE, q.trace)?;
        m.end(opts)?;
        self.call(m).ctx(format!("adding mmlink qdisc {}", q.link_id))
    }

    pub fn del_root_qdisc(&mut self, ifindex: u32) -> Result<()> {
        self.call(Msg::new(RTM_DELQDISC, 0, &tcmsg(ifindex))).ctx("deleting root qdisc")
    }
}

fn cstr(s: &str) -> String {
    format!("{s}\0")
}

fn fd_u32(fd: BorrowedFd<'_>) -> u32 {
    // Descriptors are non-negative.
    fd.as_raw_fd() as u32
}

fn rd_u16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_ne_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn rd_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_ne_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Find the ack for `seq` in one datagram. Ok(None): not in this datagram.
fn parse_ack(mut b: &[u8], seq: u32) -> Result<Option<Result<()>>> {
    while b.len() >= NLMSG_HDRLEN {
        let (Some(len), Some(ty), Some(flags), Some(s)) =
            (rd_u32(b, 0), rd_u16(b, 4), rd_u16(b, 6), rd_u32(b, 8))
        else {
            return fail("short netlink header");
        };
        let len = len as usize;
        if len < NLMSG_HDRLEN || len > b.len() {
            return fail("bad netlink message length");
        }
        let body = b.get(NLMSG_HDRLEN..len).unwrap_or_default();
        if ty == NLMSG_ERROR && s == seq {
            let Some(err) = rd_u32(body, 0).map(|e| e as i32) else { return fail("short netlink error") };
            if err == 0 {
                return Ok(Some(Ok(())));
            }
            // With NETLINK_CAP_ACK the echoed request is just its header.
            let msg = if flags & NLM_F_ACK_TLVS != 0 {
                ext_ack_msg(body.get(4 + NLMSG_HDRLEN..).unwrap_or_default())
            } else {
                None
            };
            let os = io::Error::from_raw_os_error(-err);
            let text = match msg {
                Some(m) => format!("{os} ({m})"),
                None => os.to_string(),
            };
            return Ok(Some(Err(Error(text))));
        }
        b = b.get(align4(len)..).unwrap_or_default();
    }
    Ok(None)
}

fn ext_ack_msg(mut tlvs: &[u8]) -> Option<String> {
    while tlvs.len() >= 4 {
        let len = rd_u16(tlvs, 0)? as usize;
        let ty = rd_u16(tlvs, 2)?;
        let val = tlvs.get(4..len)?;
        if ty == NLMSGERR_ATTR_MSG {
            let s = val.split(|&c| c == 0).next()?;
            return Some(String::from_utf8_lossy(s).into_owned());
        }
        tlvs = tlvs.get(align4(len.max(4))..)?;
    }
    None
}
