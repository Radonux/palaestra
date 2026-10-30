#!/bin/bash
# M0 guest probe: try to register a BPF struct_ops congestion control.
# Kernel command-line environment: probe_out probe_bpftool probe_obj
set -u
: "${probe_out:?}" "${probe_bpftool:?}" "${probe_obj:?}"
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
mount -t proc proc /proc; mount -t sysfs sys /sys
mount -t bpf bpf /sys/fs/bpf
mount -t hostfs none /mnt -o "$probe_out"
{
echo "jit_enable=$(cat /proc/sys/net/core/bpf_jit_enable 2>&1)"
echo "available_cc_before=$(cat /proc/sys/net/ipv4/tcp_available_congestion_control)"
"$probe_bpftool" -d struct_ops register "$probe_obj" /sys/fs/bpf 2>&1 | grep -iE 'error|fail|struct_ops|errno|-[0-9]+' | tail -15
echo "bpftool_rc=${PIPESTATUS[0]}"
echo "available_cc_after=$(cat /proc/sys/net/ipv4/tcp_available_congestion_control)"
} > /mnt/bpf.txt 2>&1
echo o > /proc/sysrq-trigger; sleep 5
