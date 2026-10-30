#!/bin/bash
# M0 guest probe (runs as PID 1 inside UML, host / mounted read-only via
# hostfs). Builds a 3-namespace path with netem in the middle namespace,
# optionally loads a module, runs one iperf3 bulk transfer, and writes
# results to the hostfs directory given as probe_out.
# Kernel command-line environment: probe_cc probe_rate probe_delay
# probe_secs probe_out probe_tag [probe_mod]
set -u
: "${probe_cc:?}" "${probe_rate:?}" "${probe_delay:?}" "${probe_secs:?}" "${probe_out:?}" "${probe_tag:?}"
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
mount -t proc proc /proc; mount -t sysfs sys /sys
mount -t tmpfs tmp /tmp; mount -t tmpfs run /run
mkdir -p /run/netns
mount -t hostfs none /mnt -o "$probe_out"
log="/mnt/net-$probe_tag.txt"
{
if [[ -n "${probe_mod:-}" ]]; then insmod "$probe_mod" && echo "insmod ok"; fi
for n in snd path rcv; do ip netns add $n; ip -n $n link set lo up; done
ip link add s0 netns snd type veth peer name p0 netns path
ip link add r0 netns rcv type veth peer name p1 netns path
ip -n snd addr add 10.0.0.1/24 dev s0; ip -n path addr add 10.0.0.2/24 dev p0
ip -n rcv addr add 10.0.1.1/24 dev r0; ip -n path addr add 10.0.1.2/24 dev p1
for x in "snd s0" "path p0" "path p1" "rcv r0"; do read -r ns dev <<< "$x"; ip -n "$ns" link set "$dev" up; done
ip -n snd route add default via 10.0.0.2; ip -n rcv route add default via 10.0.1.2
ip netns exec path sh -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
ip netns exec path tc qdisc add dev p1 root netem rate "$probe_rate" delay "$probe_delay" limit 100000
ip netns exec path tc qdisc add dev p0 root netem delay "$probe_delay" limit 100000
ip netns exec rcv iperf3 -s -D -1 >/dev/null
sleep 1
t0=$(date +%s.%N)
ip netns exec snd iperf3 -c 10.0.1.1 -t "$probe_secs" -C "$probe_cc" -J > "/mnt/iperf-$probe_tag.json"
t1=$(date +%s.%N)
echo "virtual_secs=$(echo "$t1 - $t0" | bc)"
if [[ -n "${probe_mod:-}" ]]; then echo "tcp_probe_hits=$(cat /sys/module/m0_probe/parameters/hits)"; fi
ip netns exec snd grep -E '^Tcp(Ext)?:' /proc/net/netstat /proc/net/snmp
ip netns exec path tc -s qdisc show dev p1
} > "$log" 2>&1
echo o > /proc/sysrq-trigger; sleep 5
