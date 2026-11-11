# LCF concepts from the literature

An LCF (latent congestion factor) is a quantity about the path that net1
estimates and net2 acts on. Its true value must be exact in the emulator;
the sender need not be able to measure it.

A basic concept is a quantity the link itself determines and the emulator
logs directly, such as capacity, the queue or a packet's delay. A
compositional concept is a fixed formula over basic concepts in a short
window (about 10 ms to 100 ms), such as the fraction of capacity in use.
Each section below gives the concepts' meanings first, then their
definitions, each copied from the source named in the last column. The
sources are listed in the appendix.

In the meaning tables:

- "Truth" says how the emulator records the value. palaestra's link qdisc
  logs every arrival, departure and drop (with its cause and its flow), the
  queue after each event, one delivery-opportunity record per millisecond,
  and the buffer limit. "Exact" means a function of those records in the
  window; "exact, per flow" also needs each packet's flow; "model" means
  the value exists only in a model the emulator does not run.
- "Sender" says whether the sender can measure the value: "yes",
  "estimate" (biased, delayed or conditional) or "no".
- "Label" is the name palaestra gives the value: a stored ground-truth
  label, or a composite the trainer computes from stored labels
  ([architecture.md](architecture.md#ground-truth-labels)). A windowed
  one exists over 100 ms under this name and over 10 ms as `<name>_10ms`.

Notation:

| Symbol | Meaning |
|---|---|
| $`C`$ | link capacity (bit/s) |
| $`y`$ | arrival rate at the link, all flows (bit/s) |
| $`\mu`$ | departure rate from the link (bit/s) |
| $`q`$ | queue (bytes or packets); $`B`$ the buffer limit |
| $`R_m`$ | round-trip propagation delay (base RTT) |
| $`N`$ | number of active flows; $`x_i`$ the rate of flow $`i`$ |
| $`[T, T+I]`$ | the window over which a value is defined |

## The path at rest

How fast the bottleneck is, how far away the receiver is, and how much the
bottleneck can hold. These change only when the path changes.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Delivery opportunity | a moment at which the link can deliver one 1500-byte packet; a trace-driven link is a list of them | exact | no | none |
| Capacity | the most the bottleneck can deliver per second in the window | exact | estimate | capacity_bps |
| Base RTT | round-trip time with empty queues | exact | estimate | base_rtt_ns, one_way_delay_ns |
| Buffer size | how much the bottleneck queue can hold before it drops | exact | estimate, after one overflow | buffer_pkts, buffer_bytes |
| Bandwidth-delay product (BDP) | data in flight when the link is full and the queue empty | exact | estimate | bdp_pkts |
| Non-congestive delay | delay that does not come from the bottleneck queue (link-layer jitter, ACK aggregation) | exact: the ACK link's queueing delay; the delay line adds none | no | ack_queue_delay_mean_ns (ACK-path queueing) |
| Bandwidth change ratio | the factor by which capacity changes at a step | exact | no | none |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Delivery opportunity | "the time at which an MTU-sized packet will be delivered"; each represents the ability to deliver 1500 bytes; an opportunity is wasted if no bytes are waiting | Mahimahi |
| Capacity | $`C(L,T,I)`$: the maximum IP-layer bits that can be sent and correctly received over link $`L`$ during $`[T, T+I]`$, divided by $`I`$ | RFC 5136 |
| Path capacity | $`C = \min_i C_i`$, the capacity of the narrow link | RFC 5136 |
| Base RTT | $`R_m`$, the round-trip propagation delay | CCAC |
| Buffer size | $`\beta`$; loss can occur only once the queue exceeds it | CCAC |
| BDP | $`C \cdot R_m`$ | CCmatic |
| Non-congestive delay | $`\mathrm{RTT}(t) = R_m + q_{\mathrm{del}}(t) + \mathrm{jitter}(t)`$, with $`0 \le \mathrm{jitter}(t) \le D`$ | CCmatic |
| Jitter bound | $`D`$: tokens that arrived $`D`$ seconds ago and were not wasted must have been used, $`S(t) \ge C\,(t-D) - W(t-D)`$ | CCAC |
| Bandwidth change ratio | the change ratio of the bandwidth, from 0.1x to 8x in its study | Mahak |

## How busy the link is

The load on the bottleneck in one window. Excess demand and idle time are
our names for simple differences.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Arrival rate | bits per second entering the bottleneck queue, all flows | exact | no | arrival_rate_bps |
| Usage (departure rate) | bits per second the link actually sent, all flows | exact | estimate, with in-network telemetry | departure_rate_bps |
| Receive rate | the observed flow's share of the usage | exact, per flow | yes | receive_rate_bps |
| Utilization (busy fraction) | the fraction of the link's capacity that was used; stops at 1 once the queue never empties | exact | estimate | busy_fraction |
| Available capacity | capacity left unused by all flows (spare capacity); never negative | exact | estimate | available_capacity_bps |
| Excess demand | arrival rate minus capacity: positive means the queue grows | exact | estimate | excess_demand_bps |
| Load factor | arrival rate over a target rate just below capacity; unlike utilization it keeps growing under overload | exact | no | load_factor |
| Idle time | the fraction of the window in which the link had nothing to send | exact | no | none (1 minus busy fraction) |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Arrival rate | $`y(t)`$, the measured input traffic rate over the last update interval | RCP |
| Usage | $`\mathrm{Used}(L,T,I)`$: the IP-layer bits from any source correctly received over $`L`$ during $`[T, T+I]`$, divided by $`I`$ | RFC 5136 |
| Departure rate | $`\mathrm{txRate} = \Delta\mathrm{txBytes} / \Delta t`$ between two telemetry records | HPCC |
| Utilization | $`U(L,T,I) = \mathrm{Used}(L,T,I) / C(L,T,I)`$ | RFC 5136 |
| Utilization, time average | $`\bar u(t-\tau, t) = \frac{1}{\tau}\int_{t-\tau}^{t} u(x)\,dx`$, with $`u(x) \in \{0,1\}`$ the instantaneous busy indicator and $`\tau`$ the averaging timescale | Prasad 2003 |
| Available capacity | $`A(L,T,I) = C(L,T,I)\,\bigl(1 - U(L,T,I)\bigr)`$; the tight link is the one with the least | RFC 5136 |
| Available capacity, fluid form | $`A_i(t,T) = \frac{1}{T}\int_t^{t+T} \bigl(C_i - \lambda_i(x)\bigr)\,dx`$, $`\lambda_i`$ the cross-traffic rate | Spruce |
| Excess demand | $`y - C`$; XCP's spare bandwidth is the same difference with the opposite sign | XCP |
| Load factor | $`z = y / C_{\mathrm{target}}`$, with $`C_{\mathrm{target}}`$ 85% to 95% of link bandwidth | OSU scheme (ATM survey) |
| Link idle event | the queue becomes empty or the link goes idle | BLUE |

## The queue

How big the bottleneck queue is, how long packets wait in it, whether it is
a standing queue or a passing burst, and where it is heading.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Queue | data waiting in the bottleneck buffer now | exact | estimate | queue_pkts, queue_bytes |
| Free buffer | buffer space left before drops start | exact | estimate | free_buffer_pkts |
| Buffer fill | the fraction of the buffer in use | exact | estimate | buffer_fill |
| Sojourn time | how long a packet waited in the queue | exact | estimate (RTT minus base RTT) | queue_delay_ns, queue_delay_mean_ns |
| Standing queue | the part of the queue that does not drain within a round trip: its minimum over a recent window | exact | estimate | queue_min_pkts |
| Extra data in the network | data the flow keeps queued beyond what the path holds with an empty queue | exact, per flow | estimate | flow_queue_pkts |
| Average queue | a smoothed queue | exact | no | none |
| Queue growth rate | how fast the queue grows or drains, in seconds of delay per second | exact | estimate (RTT gradient) | none (load_factor minus 1) |
| Little's-law delay | the queue divided by the departure rate: how long a packet arriving now will wait | exact | no | littles_delay_ns |
| Normalized inflight | how full the path is: 1 when the link is busy and its queue empty, plus the queue as a fraction of the BDP | exact | estimate, with telemetry | normalized_inflight |
| Normalized power | (queue plus BDP) over BDP, times arrival rate over capacity: 1 when the link is just full; it rises with the queue, and with the arrival rate before the queue has built | exact | estimate | normalized_power |
| Classic power | throughput over delay; highest at the knee, where the link is full and the queue still empty | exact | estimate | none |
| Virtual queue | the queue a slightly slower link would have; it fills before the real queue does | exact | no | none |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Queue | $`Q(t) = A(t) - L(t) - S(t)`$ over cumulative arrivals, losses and service | CCAC |
| Free buffer | $`B - q`$ (expliCCit's LCF) | palaestra |
| Buffer fill | $`q / B`$ | palaestra |
| Sojourn time | "the time a packet departs the buffer minus the time the packet arrived at the buffer" | CoDel (RFC 8289) |
| Standing queue | the minimum queue seen by an arriving packet during the last propagation delay | XCP |
| Standing queue, by delay | the sojourn time has stayed above TARGET = 5 ms for INTERVAL = 100 ms | CoDel (RFC 8289) |
| Standing RTT | the smallest RTT over the last $`\mathrm{srtt}/2`$; Copa's queueing delay is $`d_q = \mathrm{RTT}_{\mathrm{standing}} - \mathrm{RTT}_{\min}`$, with $`\mathrm{RTT}_{\min}`$ the smallest RTT over the last 10 s or since the flow started | Copa |
| Extra data in the network | $`\mathrm{Diff} = \mathrm{Expected} - \mathrm{Actual}`$, with $`\mathrm{Expected} = \mathrm{WindowSize} / \mathrm{BaseRTT}`$ | Vegas |
| Average queue | $`\bar q \leftarrow (1 - w_q)\,\bar q + w_q\,q`$ on each arrival, $`w_q = 0.002`$ in most of its simulations (formula recovered from a PDF with damaged symbols) | RED |
| Average queue over a busy cycle | the area under $`q(t)`$ since the start of the previous busy period, divided by the elapsed time | DECbit |
| Queue growth rate | $`dq/dt = (y - C)/C`$, with the queue in seconds | TIMELY |
| Little's-law delay | $`\hat d = q / \mu`$ | PIE (RFC 8033) |
| Normalized inflight | $`U_j = q_j / (C_j T) + \mathrm{txRate}_j / C_j`$ for link $`j`$ with capacity $`C_j`$ (written $`B_j`$ in the source) and base RTT $`T`$ | HPCC |
| Normalized power | $`\Gamma(t) = \bigl(q(t) + b\,\tau\bigr)\,\lambda(t)`$, normalized as $`\Gamma / (b^2 \tau)`$, with $`b`$ the bottleneck bandwidth, $`\tau`$ the base RTT and $`\lambda`$ the arrival rate; equivalently $`\bigl(1 + q/(b\tau)\bigr)\,\lambda/b`$ (our algebra) | PowerTCP |
| Classic power | $`P = \mathrm{throughput}^{\alpha} / \mathrm{response\ time}`$, normally $`\alpha = 1`$ | Jain-Ramakrishnan |
| Queueing power | $`P(\rho) = \rho / \bigl(\mu T(\rho)\bigr)`$, $`\rho`$ the utilization and $`\mu`$ the service rate; for M/M/1 it peaks at $`\rho = 0.5`$ | Kleinrock |
| Virtual queue | on each arrival $`VQ \leftarrow \max\bigl(VQ - \tilde C\,(t - s), 0\bigr)`$ and the packet is marked if $`VQ + b > B`$ ($`b`$ the packet size, $`s`$ the previous arrival); the virtual capacity follows $`\dot{\tilde C} = \alpha\,(\gamma C - \lambda)`$, $`\gamma`$ the target utilization | AVQ |

## Delay and loss at the endpoints

What the endpoints see of the queue: RTT and its smoothing, delay trend and
variation, losses and marks.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Smoothed RTT (sRTT) | the sender's running average of RTT samples, with a running measure of their variation | exact (sender state, recorded per ACK) | yes | none |
| Minimum RTT | the smallest RTT seen; equals the base RTT once the queue has emptied | exact (sender state) | yes | none |
| One-way delay | time from sending a packet to its arrival | exact | partly: needs synchronized clocks | one_way_delay_ns (the delay line) |
| Delay variation | the difference in delay between two packets; jitter is its smoothed size | exact | partly | none |
| Delay gradient | whether delay is rising or falling, relative to its level | exact | yes | none |
| Latency ratio | current RTT over base RTT | exact | yes | rtt_ratio (on the true delays) |
| Sending ratio | packets sent over packets delivered: above 1 means a queue or losses build | exact, per flow | yes | sending_ratio (at the link) |
| Loss | whether a packet was dropped, with the cause | exact | yes, late and ambiguous | drop_fraction, random_drop_fraction, random_loss_rate |
| Congestion mark | an ECN mark a router sets instead of dropping | exact if the emulated queue marks | yes | none |
| Marked fraction | the fraction of packets that saw the queue above a threshold | exact | yes, with marking | none |
| Loss-throughput relation | the rate a TCP flow reaches at a given loss rate and RTT | exact | yes | none |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Smoothed RTT | $`\mathrm{RTTVAR} \leftarrow (1-\beta)\,\mathrm{RTTVAR} + \beta\,\lvert \mathrm{SRTT} - R' \rvert`$, then $`\mathrm{SRTT} \leftarrow (1-\alpha)\,\mathrm{SRTT} + \alpha R'`$, with $`\alpha = 1/8`$ and $`\beta = 1/4`$ | RFC 2988 |
| Minimum RTT | BaseRTT, "the minimum of all measured round trip times" | Vegas |
| One-way delay | $`dT`$: the first bit leaves at wire time $`T`$ and the last bit arrives at $`T + dT`$; undefined if not received within $`T_{\max}`$ | RFC 7679 |
| Delay variation | $`\mathrm{ipdv} = dT_2 - dT_1`$ for a selected packet pair | RFC 3393 |
| Jitter | an exponential filter of $`\lvert \mathrm{ipdv} \rvert`$ with gain 1/16 | RFC 3393 |
| Delay gradient | $`\mathrm{rtt\_diff} \leftarrow (1-\alpha)\,\mathrm{rtt\_diff} + \alpha\,(\mathrm{new\_rtt} - \mathrm{prev\_rtt})`$; normalized gradient $`= \mathrm{rtt\_diff} / \mathrm{minRTT}`$ | TIMELY |
| Normalized delay gradient | $`\mathrm{NDG} = \frac{dD/dW}{D/W} \approx \frac{D - D_{\mathrm{old}}}{D + D_{\mathrm{old}}} \cdot \frac{W + W_{\mathrm{old}}}{W - W_{\mathrm{old}}}`$, positive above the knee | Jain 1989 |
| Latency gradient | the derivative of latency with respect to time over a monitor interval | Aurora |
| Latency ratio | a monitor interval's mean latency over the smallest such mean so far | Aurora |
| RTT ratio | the latest RTT over the minimum RTT of the connection | Remy |
| Sending ratio | packets sent over packets acknowledged by the receiver | Aurora |
| Loss | 0 if the packet arrives within $`T_{\max}`$, 1 if not; the loss ratio is the mean over a stream | RFC 7680 |
| Congestion mark | the CE codepoint, set "only if the router would otherwise have dropped the packet", from the average queue | RFC 3168 |
| Marked fraction | $`F`$, the fraction of packets marked in the last window of data, smoothed as $`\alpha \leftarrow (1-g)\,\alpha + g F`$ with $`g = 1/16`$; $`\alpha`$ estimates the probability that the queue exceeds $`K`$ | DCTCP |
| Loss-throughput relation | $`\mathrm{BW} = (\mathrm{MSS}/\mathrm{RTT}) \cdot k / \sqrt{p}`$ for loss rate $`p`$, $`k = \sqrt{3/2}`$ (the source calls it $`C`$) | Mathis |

## The other flows

Who else is on the link and how the capacity is split.

| Concept | What it means | Truth | Sender | Label |
|---|---|---|---|---|
| Cross-traffic rate | arrival rate of all other flows | exact, per flow | estimate | cross_arrival_rate_bps, cross_traffic_share |
| Active flows | how many flows have packets at the bottleneck in the window | exact, per flow | no | active_flows |
| Fair share | capacity divided equally among the active flows | exact, per flow | no | fair_share_bps |
| Max-min share | the fair share after flows that need less have been served | exact, per flow | no | none |
| Fairness index | how evenly the link is split: 1 for equal shares, $`k/n`$ when $`k`$ of $`n`$ flows share equally and the rest get nothing | exact, per flow | no | none |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Cross-traffic rate | $`r_i`$, the average rate of cross traffic that enters the path at link $`i`$ | Dovrolis 2001 |
| Cross-traffic estimate | $`\hat z = \mu S / R - S`$, from the flow's send rate $`S`$ and receive rate $`R`$ and the bottleneck rate $`\mu`$ | Nimbus |
| Active flows | $`N(t)`$, the number of ongoing flows | RCP |
| Flow-count estimate | $`\hat N(t) = C / R(t - T)`$, $`R`$ the fair rate the router last computed | RCP |
| Fair share | $`C / N(t)`$, the processor-sharing rate | RCP |
| Fair share, with a target | target rate over the number of active sources | OSU scheme (ATM survey) |
| Max-min share | $`\bigl(C - \sum_{\mathrm{underloading}} x_i\bigr) / (N - N_{\mathrm{underloading}})`$, iterated until the set of underloading flows stops changing | MIT scheme (ATM survey) |
| Fairness index | $`F = \bigl(\sum_i x_i\bigr)^2 / \bigl(n \sum_i x_i^2\bigr)`$ for $`n`$ flows | Jain-Chiu-Hawe |

## The cause of congestion

Why the queue builds rather than how much. A sender needs hundreds of
milliseconds to seconds to estimate these; in the emulator each is a fact
of the episode.

| Concept | What it means | Window at the sender | Truth |
|---|---|---|---|
| Elasticity | whether the competing traffic backs off when we send more (a TCP flow does, a fixed-rate stream does not) | about 5 s | exact: each competitor's type is known |
| Shared bottleneck | whether two flows wait in the same queue | 350 ms intervals, over 30 to 50 of them; about 1 s for correlation tests | exact: known from the topology |
| Self-induced congestion | whether this flow filled a previously empty queue itself, rather than meeting a queue other traffic built | slow start | exact: the queue before the flow ramped up |
| Loss cause | whether a loss came from a full queue or from a random error | the delay history around each loss | exact: each drop record names its cause |
| Loss event rate | how often loss events occur, counting several losses within one round trip as one event | eight loss intervals | exact |
| ACK compression | ACKs bunched together by a queue on the return path, which breaks self-clocking | per burst | exact: both directions are recorded |

Definitions:

| Concept | Definition | Source |
|---|---|---|
| Elasticity | pulse the sending rate at $`f_p = 5`$ Hz and compute $`\eta = \lvert \mathrm{FFT}_{\hat z}(f_p) \rvert \,/\, \max_{f \in (f_p, 2 f_p)} \lvert \mathrm{FFT}_{\hat z}(f) \rvert`$ over the cross-traffic estimate $`\hat z`$; inelastic if $`\eta < 2`$ | Nimbus |
| Shared bottleneck | per flow, from one-way delay and loss: skew_est (samples below minus above the mean delay, over the count), var_est (mean absolute deviation), freq_est (significant crossings of the mean, over the number of intervals) and pkt_loss; base interval 350 ms, 30 intervals for the mean, skew_est and var_est, 50 for freq_est and pkt_loss | RFC 8382 |
| Shared congestion test | the delay (or loss) correlation between two flows' packets compared with that within one flow; shared if the cross-measure exceeds the auto-measure | Rubenstein 2002 |
| Self-induced congestion | during slow start, $`\mathrm{NormDiff} = (\mathrm{RTT}_{\max} - \mathrm{RTT}_{\min}) / \mathrm{RTT}_{\max}`$ and the coefficient of variation of RTT; labelled self-induced when slow-start throughput exceeds 0.8 of capacity | Congestion signatures |
| Loss cause | the Biaz rule (the gap around a loss against multiples of the minimum inter-arrival time), the Spike rule (relative one-way delay against thresholds between its minimum and maximum) and ZigZag (loss count and delay against its mean and deviation); some constants unreadable in the copy | Cen 2003 |
| Loss event rate | a loss event is one or more losses within one RTT; $`p = 1 / \bar I`$, $`\bar I`$ the weighted mean of the last eight loss intervals | TFRC (RFC 5348) |
| ACK compression | ACKs that meet a nonempty queue leave it spaced by an ACK's transmission time instead of a data packet's | Zhang-Shenker-Clark |

## Concepts defined only through a model

These exist only in a model of the network, so the emulator cannot record
them directly.

| Concept | What it means | Definition | Source |
|---|---|---|---|
| Link price (REM) | a running congestion cost that rises while demand exceeds capacity or the queue exceeds its target | $`p_l(t+1) = \bigl[p_l(t) + g\,\bigl(a_l\,(b_l(t) - b_l^*) + x_l(t) - c_l(t)\bigr)\bigr]^+`$ | REM |
| Link price (dual) | the same idea with the rate term only; a path's price is the sum of its links' prices | $`p_l(t+1) = \bigl[p_l(t) + \gamma\,(x^l - c_l)\bigr]^+`$ | Low-Lapsley |
| Proportional fairness | the allocation that maximizes the sum of log rates under the capacity limits | $`\sum_r (x_r^* - x_r)/x_r \le 0`$ for every feasible $`x^*`$; the optimum of $`\sum_r w_r \log x_r`$ subject to $`Ax \le C`$ | Kelly 1998 |
| Knee and cliff | the knee is the load past which throughput hardly grows but delay does; the cliff is where throughput collapses | "the point after which the increase in the throughput is small, but a significant increase in the response time results"; the cliff is the point of congestion collapse | Jain-Ramakrishnan |
| Bottleneck link of a flow | the link whose capacity limits the flow's rate | flow $`f`$ is bottlenecked at link $`l`$ if it traverses $`l`$ and $`dr_f/dc_l \ne 0`$ | QTBS |
| Link-rate bounds | the range of link rates consistent with the ACKs seen so far | from an ACK rate $`r`$ over an interval $`T`$, $`rT \in [C(T - D), C(T + D)]`$ when the sending rate is at least $`C`$ | CCmatic |
| Link-rate forecast | a probability distribution over the next link rate, updated from arrivals | a Poisson process whose rate follows Brownian motion; a 5th-percentile forecast of the bytes delivered in the next 160 ms | Sprout |

## Ad-hoc concepts from concept-based explainers

Agua names eight congestion control concepts without formulas; a concept's
value there is the similarity between text embeddings of an LLM-written
description of the state and of the concept's name. The same group's
earlier paper (Patel 2024) defines concepts for adaptive bitrate only. The
closed forms in the second column are our proposal, built from the
definitions above.

| Concept (Agua) | Closed form we propose | Built from |
|---|---|---|
| Increasing packet loss, decreasing packet loss | the sign and size of the change in loss ratio between consecutive windows | RFC 7680 |
| Rapidly increasing latency, rapidly decreasing latency | the queue growth rate, $`(y - C)/C`$, above or below a threshold | TIMELY |
| Low network utilization, high network utilization | utilization $`U`$ below or above a threshold | RFC 5136 |
| Stable network conditions, volatile network conditions | the coefficient of variation of capacity, or of available capacity, over a window | ours |

## Appendix: sources

Each name used above, the paper it stands for, and the text the
definitions were copied from.

| Name | Paper | Definitions copied from |
|---|---|---|
| Agua | Patel et al., Agua: A Concept-Based Explainer for Learning-Enabled Systems (2025), https://doi.org/10.1145/3718958.3754341 | https://ina.kaist.ac.kr/assets/bibliography/Agua.pdf |
| ATM survey | Jain, Congestion Control and Traffic Management in ATM Networks: Recent Advances and A Survey (1998), https://arxiv.org/abs/cs/9809085 | https://arxiv.org/pdf/cs/9809085v1 (the 1996 draft; source of the OSU and MIT schemes cited above) |
| Aurora | Jay et al., Internet Congestion Control via Deep Reinforcement Learning (2018), https://arxiv.org/abs/1810.03259 | https://arxiv.org/pdf/1810.03259v3 |
| AVQ | Kunniyur and Srikant, Analysis and design of an adaptive virtual queue (AVQ) algorithm for active queue management (2001), https://doi.org/10.1145/964723.383069 | https://typeset.io/pdf/analysis-and-design-of-an-adaptive-virtual-queue-avq-17o3myyejf.pdf |
| BLUE | Feng et al., The BLUE active queue management algorithms (2002), https://doi.org/10.1109/tnet.2002.801399 | http://www.thefengs.com/wuchang/blue/CSE-TR-387-99.pdf (definitions copied from the 1999 technical report version) |
| CCAC | Arun et al., Toward formally verifying congestion control behavior (2021), https://doi.org/10.1145/3452296.3472912 | https://saeed.github.io/files/ccac-sigcomm21.pdf |
| CCmatic | Agarwal et al., Towards Provably Performant Congestion Control (NSDI 2024) | https://www.usenix.org/system/files/nsdi24-agarwal-anup.pdf |
| Cen 2003 | Cen et al., End-to-end differentiation of congestion and wireless losses (2003), https://doi.org/10.1109/tnet.2003.818187 | https://typeset.io/pdf/end-to-end-differentiation-of-congestion-and-wireless-losses-2u8o936umo.pdf (definitions copied from the submission draft) |
| CoDel (RFC 8289) | Nichols and Jacobson, Controlled Delay Active Queue Management (2018), https://doi.org/10.17487/rfc8289 | https://www.rfc-editor.org/rfc/rfc8289.txt |
| Congestion signatures | Sundaresan et al., TCP Congestion Signatures (2018), https://doi.org/10.1145/3232755.3232760 | https://users.caida.org/~amogh/papers/signatures-IMC17.pdf (definitions copied from the IMC 2017 paper of the same title) |
| Copa | Arun and Balakrishnan, Copa: Practical Delay-Based Congestion Control for the Internet (NSDI 2018) | https://www.usenix.org/system/files/conference/nsdi18/nsdi18-arun.pdf |
| DCTCP | Alizadeh et al., Data center TCP (DCTCP) (2010), https://doi.org/10.1145/1851182.1851192 | https://people.csail.mit.edu/alizadeh/papers/dctcp-sigcomm10.pdf |
| DECbit | Ramakrishnan and Jain, A binary feedback scheme for congestion avoidance in computer networks (1990), https://doi.org/10.1145/78952.78955 | https://arxiv.org/pdf/cs/9809094 (definitions copied from the DEC technical report version) |
| Dovrolis 2001 | Dovrolis et al., What do packet dispersion techniques measure? (2001), https://doi.org/10.1109/infcom.2001.916282 | https://homepage.iis.sinica.edu.tw/~cclljj/teaching/06S-Advanced_Computer_Networks/papers/4_pathrate.pdf |
| HPCC | Li et al., HPCC: High precision congestion control (2019), https://doi.org/10.1145/3341302.3342085 | https://liyuliang001.github.io/publications/hpcc.pdf |
| Jain 1989 | Jain, A delay-based approach for congestion avoidance in interconnected heterogeneous computer networks (1989), https://doi.org/10.1145/74681.74686 | https://arxiv.org/pdf/cs/9809093 (definitions copied from the DEC report version) |
| Jain-Chiu-Hawe | Jain et al., A Quantitative Measure Of Fairness And Discrimination For Resource Allocation In Shared Computer Systems (1998), https://arxiv.org/abs/cs/9809099 | https://arxiv.org/pdf/cs/9809099 (a scanned DEC technical report) |
| Jain-Ramakrishnan | Jain and Ramakrishnan, Congestion Avoidance in Computer Networks with a Connectionless Network Layer, Part I: Concepts, Goals and Methodology (1998), https://arxiv.org/abs/cs/9809095 | https://arxiv.org/pdf/cs/9809095v1 |
| Kelly 1998 | Kelly et al., Rate control for communication networks: shadow prices, proportional fairness and stability (1998), https://doi.org/10.1038/sj.jors.2600523 | http://www.statslab.cam.ac.uk/~frank/rate.pdf |
| Kleinrock | Kleinrock, Internet congestion control using the power metric: Keep the pipe just full, but no fuller (2018), https://doi.org/10.1016/j.adhoc.2018.05.015 | https://www.lk.cs.ucla.edu/data/files/Kleinrock/Internet%20Congestion%20Control%20Using%20the%20Power%20Metric-Keep%20the%20Pipe%20Just%20Full,%20But%20No%20Fuller%20July%202018.pdf (definitions copied from the author's 2018 preprint) |
| Low-Lapsley | Low and Lapsely, Optimization flow control. I. Basic algorithm and convergence (1999), https://doi.org/10.1109/90.811451 | https://www.ee.columbia.edu/~tbma/files/GameCourse/Low_Optimization.pdf |
| Mahak | Pazhooheshy et al., Harnessing ML For Network Protocol Assessment: A Congestion Control Use Case (2023), https://doi.org/10.1145/3626111.3628182 | https://conferences.sigcomm.org/hotnets/2023/papers/hotnets23_pazhooheshy.pdf |
| Mahimahi | Netravali et al., Mahimahi: Accurate Record-and-Replay for HTTP (USENIX ATC 2015) | https://www.cs.princeton.edu/~ravian/publications/mahimahi_atc.pdf |
| Mathis | Mathis et al., The macroscopic behavior of the TCP congestion avoidance algorithm (1997), https://doi.org/10.1145/263932.264023 | https://courses.cs.duke.edu/fall24/compsci514/readings/mathis-tcpmodel-ccr97.pdf |
| Nimbus | Goyal et al., Elasticity detection: A building block for internet congestion control (2022), https://doi.org/10.1145/3544216.3544221 | https://arxiv.org/pdf/1802.08730v4 (definitions copied from arXiv v4 (2020) of the same work) |
| palaestra | this repository's ground-truth labels, which implement expliCCit's LCFs | [architecture.md](architecture.md#ground-truth-labels) |
| Patel 2024 | Patel et al., Toward Trustworthy Learning-Enabled Systems with Concept-Based Explanations (2024), https://doi.org/10.1145/3696348.3696894 | https://conferences.sigcomm.org/hotnets/2024/papers/hotnets24-366.pdf |
| PIE (RFC 8033) | Pan et al., Proportional Integral Controller Enhanced (PIE): A Lightweight Control Scheme to Address the Bufferbloat Problem (2017), https://doi.org/10.17487/rfc8033 | https://www.rfc-editor.org/rfc/rfc8033.txt |
| PowerTCP | Addanki et al., PowerTCP: Pushing the Performance Limits of Datacenter Networks (2021), https://arxiv.org/abs/2112.14309 | https://arxiv.org/pdf/2112.14309v1 |
| Prasad 2003 | Prasad et al., Bandwidth estimation: metrics, measurement techniques, and tools (2003), https://doi.org/10.1109/mnet.2003.1248658 | https://www.caida.org/catalog/papers/2003_bwestmetrics/bwestmetrics.pdf |
| QTBS | Ros-Giralt et al., A Quantitative Theory of Bottleneck Structures for Data Networks (2022), https://arxiv.org/abs/2210.03534 | https://arxiv.org/pdf/2210.03534v1  |
| RCP | Dukkipati et al., Processor Sharing Flows in the Internet (2005), https://doi.org/10.1007/11499169_22 | http://yuba.stanford.edu/~nanditad/RCP-IWQoS.pdf |
| RED | Floyd and Jacobson, Random early detection gateways for congestion avoidance (1993), https://doi.org/10.1109/90.251892 | https://www.icir.org/floyd/papers/early.twocolumn.pdf |
| REM | Athuraliya et al., REM: active queue management (2001), https://doi.org/10.1109/65.923940 | https://typeset.io/pdf/rem-active-queue-management-1536n62j9p.pdf |
| Remy | Winstein and Balakrishnan, TCP ex machina: Computer-generated congestion control (2013), https://doi.org/10.1145/2486001.2486020 | https://stuff.mit.edu/afs/sipb/project/remy/TCPexMachina.pdf |
| RFC 2988 | Paxson and Allman, Computing TCP's Retransmission Timer (2000), https://doi.org/10.17487/rfc2988 | https://www.rfc-editor.org/rfc/rfc2988.txt |
| RFC 3168 | Ramakrishnan et al., The Addition of Explicit Congestion Notification (ECN) to IP (2001), https://doi.org/10.17487/rfc3168 | https://www.rfc-editor.org/rfc/rfc3168.txt |
| RFC 3393 | Demichelis and Chimento, IP Packet Delay Variation Metric for IP Performance Metrics (IPPM) (2002), https://doi.org/10.17487/rfc3393 | https://www.rfc-editor.org/rfc/rfc3393.txt |
| RFC 5136 | Chimento and Ishac, Defining Network Capacity (2008), https://doi.org/10.17487/rfc5136 | https://www.rfc-editor.org/rfc/rfc5136.txt |
| RFC 7679 | Almes et al., A One-Way Delay Metric for IP Performance Metrics (IPPM) (2016), https://doi.org/10.17487/rfc7679 | https://www.rfc-editor.org/rfc/rfc7679.txt |
| RFC 7680 | Almes et al., A One-Way Loss Metric for IP Performance Metrics (IPPM) (2016), https://doi.org/10.17487/rfc7680 | https://www.rfc-editor.org/rfc/rfc7680.txt |
| RFC 8382 | Ferlin et al., Shared Bottleneck Detection for Coupled Congestion Control for RTP Media (2018), https://doi.org/10.17487/rfc8382 | https://www.rfc-editor.org/rfc/rfc8382.txt |
| Rubenstein 2002 | Rubenstein et al., Detecting shared congestion of flows via end-to-end measurement (2002), https://doi.org/10.1109/tnet.2002.1012369 | https://www.eecis.udel.edu/~mills/fractal/01012369.pdf |
| Sprout | Winstein et al., Stochastic Forecasts Achieve High Throughput and Low Delay over Cellular Networks (NSDI 2013) | https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final113.pdf |
| Spruce | Strauss et al., A measurement study of available bandwidth estimation tools (2003), https://doi.org/10.1145/948205.948211 | https://pdos.csail.mit.edu/~jastr/papers/imc03-spruce.pdf |
| TFRC (RFC 5348) | Floyd et al., TCP Friendly Rate Control (TFRC): Protocol Specification (2008), https://doi.org/10.17487/rfc5348 | https://www.rfc-editor.org/rfc/rfc5348.txt |
| TIMELY | Mittal et al., TIMELY: RTT-based Congestion Control for the Datacenter (2015), https://doi.org/10.1145/2785956.2787510 | https://conferences.sigcomm.org/sigcomm/2015/pdf/papers/p537.pdf |
| Vegas | Brakmo and Peterson, TCP Vegas: end to end congestion avoidance on a global Internet (1995), https://doi.org/10.1109/49.464716 | http://www.cs.princeton.edu/courses/archive/fall06/cos561/papers/vegas.pdf |
| XCP | Katabi et al., Congestion control for high bandwidth-delay product networks (2002), https://doi.org/10.1145/964725.633035 | http://conferences.sigcomm.org/sigcomm/2002/papers/xcp.pdf |
| Zhang-Shenker-Clark | Zhang et al., Observations on the dynamics of a congestion control algorithm: The effects of two-way traffic (1991), https://doi.org/10.1145/115992.116006 | https://groups.csail.mit.edu/ana/Publications/PubPDFs/Observations%20on%20the%20Dynamics%20of%20a%20Congestion%20Control%20Algorithm.pdf |
