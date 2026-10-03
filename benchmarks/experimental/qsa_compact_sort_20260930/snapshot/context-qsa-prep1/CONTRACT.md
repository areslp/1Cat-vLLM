This is a finite per-context QSA description for human review, not W2 performance
admission. A0/B/A2 use unchanged pure off/on library identities controlled by a
separate reviewed service controller. Service, actual startup, fresh KV capacity,
and execution approval remain PENDING here. The package sends no requests during
preparation. The old run4 NO_GO and the original 0.5%/CI thresholds stay unchanged.

The 19 rows are 512/2048/8192/32768/65536 tokens at c1/c4/c8; 131072 at c1/c4;
261632 at c1/c2. Each arm runs each row twice, in the same ascending row/ordinal
order, with 256 greedy outputs, seed1729, ignore_eos, homogeneous simultaneous
release after every client READY. No mixed arrivals, replacements, retry, added
samples or threshold changes. c1/c2 are original fallback compatibility rows;
c4/c8 have static candidate routes only if actual FULL consumer shapes apply.
Startup graph binding does not establish dynamic activation. Client-common
window evidence does not alone prove a particular GPU dispatch consumer count.

512 stays cold. Other rows predeclare one sequential cold1-token prime per
measured request, with the same request-specific salt/prompt as its measured
256-token request; each prime completes and queue drains before proceeding.
There are 438 measured requests plus360 primes, 798HTTP and112488 output tokens.
Prime TTFT is cold sequential c1 evidence, distinct from concurrent measured
TTFT. Priming uses the original native cache without resets or changed semantics.
Main actual hits are retained even when zero: Mamba cache alignment can prevent
reusable state. Aggregate hits do not prove every request warm. Label these
native-primed, not allwarm. Prime exact terminal/cache/drain and fresh four-rank
E7 equality are strict; prime hook_steps may be0. Main inherited native/E7 guard
remains strict. The known lazy HTTP/ITL extra series are not this guard's native
counter families: missing first-group extra observations remain UNAVAILABLE,
without fabricated zero, and all native raw metrics remain available.

Tokens through8K use exact existing8192-token source prefixes; 32K uses its
actual historical source. Above32K are NEW synthetic periodic tiled32K inputs,
not historical fixtures or natural long documents. Bodies and per-request salts
are identical across matched arms except request_id/header identity. Each
group/repeat/request salt is unique, so priming is deliberate and other reuse is
excluded. Frozen packet SHA and complete SSE/usage/finish/DONE/token checks bind
every request. 128K/c8 andnear/c4,c8 exceed the conservative block-rounded nominal
KV663816 budget and are excluded before generation. Fresh actual capacity still
must agree. near261632 leaves512 tokens below model max262144, including256 output
and4 speculative reserve; this is feasibility admission, not a completion guarantee.

Report TTFT, end-to-end makespan/token throughput, full decode client intervals,
and common interval from max(first positive receipt) to min(last positive receipt).
Only whole positive-to-positive intervals within that common window contribute
their actual right-chunk token count. Main decode comparison is pooled
1000*sum(covered seconds)/sum(actual tokens), not a chunk gap or GPU step. Full
window token throughput remains descriptive and boundary biased. Empty windows
are NO_COMMON_WINDOW; any member with no whole interval is INSUFFICIENT_COMMON_EVENTS.
Neither case is discarded or replaced. c1 uses the same algorithm. Two repetitions
give paired descriptions and both raw values, no CI or statistical acceptance.
API output token equality compares A0/A2 and B separately; B-only difference
STOPs, and AA self-variation remains inconclusive. Prior finite numerical/logits
evidence is not expanded to these contexts.

Each request has a real external process deadline600s; measured group660s starts
after priming. Priming block cap=c*(600+30), but all work shares the arm7200s hard
deadline. Slow paths can STOP without finishing the inventory; budgets are not
time predictions. CPU parent/children14 and async scanner42, quota200%, memory512MiB,
Swap0, CVD empty for clients. Single packet4MiB/group8MiB bounds are frozen.
Index contains only small pins; groups load lazily. Aggregate8GiB measured STOP
uses the existing scanner in a separate process, stale5s; no synchronous scans
inside request pumping. Failure preserves partial bytes, actual child exits,
close/reap evidence and cleanup errors. Root owns model stops and full53/17/24
restoration even on failure. No host, GPU or model execution occurred in this package.
