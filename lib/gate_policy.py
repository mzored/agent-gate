"""The admission rule: whether a queued request may start now, and how many CPUs its runners get.

T is the token pool (logical CPUs unless AGENT_GATE_TOKENS says otherwise). A request reserves k
tokens, its claim. Tokens decide who starts; the runners' fan-out is decided apart, so a run alone
on an idle machine uses the whole machine while its reservation stays small enough for others.

Who starts:

- Critical memory pressure holds every request.
- A run starts at once when no other run holds tokens.
- Beside other runs, a claim must fit: the runs together hold at most T tokens, or T plus the
  headroom (AGENT_GATE_SMALL_HEADROOM) when the claim is 1 token.
- Beside other runs, measured swap-out over AGENT_GATE_MAX_SWAPOUT_MBPS holds every request. A claim
  above 1 token also waits while free memory is below AGENT_GATE_MIN_FREE_PCT, and, while free memory
  is within AGENT_GATE_RAMP_MARGIN_PCT of that minimum, until the previous admission is
  AGENT_GATE_RAMP_SECONDS old. A 1-token run is small, so only that evidence of harm holds it, not
  the prediction.
- The queue is FIFO, but a later request may start first while its claim fits and every request
  ahead waits for tokens alone: a request ahead that fits goes first. Once a request ahead has
  waited AGENT_GATE_BACKFILL_SECONDS, only a 1-token run into the headroom passes it, and only while
  runs outside the headroom hold all T tokens, so the aged request waits for no run that starts
  after it.

Fan-out (`Decision.workers`): a default claim (no `-c`) gets the CPUs no other run's runners
use, max(k, T - the other runs' workers), while memory has room for more work; otherwise, and for
an explicit `-c`, its claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Protocol, Sequence

#: kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
CRITICAL = 4
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}


class Limits(Protocol):
    """The tuning the rule reads (agent_gate.Settings)."""

    tokens: int
    headroom: int
    min_free: float
    ramp: float
    ramp_margin: float
    max_swapout: float
    backfill: float


def number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def run_workers(job: Mapping[str, object]) -> float:
    """The CPUs a live run's runners may use: its granted workers, or its claim."""
    return number(job.get("workers"), number(job.get("k")))


@dataclass(frozen=True)
class Snapshot:
    """The live runs and the host's memory signal at one moment."""

    jobs: Sequence[Mapping[str, object]]
    pressure: Optional[int]
    free: Optional[int]
    swapout: Optional[float]
    #: Seconds since the last admission on this host.
    since_admit: float

    @property
    def used(self) -> float:
        """Tokens held by live runs."""
        return sum(number(job.get("k")) for job in self.jobs)

    @property
    def in_headroom(self) -> float:
        """Tokens of `used` held by runs admitted into the headroom."""
        return sum(number(job.get("k")) for job in self.jobs if job.get("headroom") is True)

    @property
    def outside_headroom(self) -> float:
        return self.used - self.in_headroom

    @property
    def workers(self) -> float:
        """CPUs the live runs' runners may use together."""
        return sum(run_workers(job) for job in self.jobs)

    def memory_text(self) -> str:
        if self.free is None:
            return "memory signal unavailable"
        level = "pressure unreadable" if self.pressure is None else PRESSURE_NAMES.get(self.pressure, self.pressure)
        swap = "" if self.swapout is None else f", swapping out {self.swapout:.1f} MiB/s"
        return f"memory {level}, {self.free}% free{swap}"


@dataclass(frozen=True)
class Decision:
    """`code` is "start" when the request may start now; otherwise why not, and `reason` says it in words."""

    code: str
    reason: Optional[str] = None
    #: The fan-out its runners get, when it starts.
    workers: int = 0
    #: It starts only thanks to the 1-token headroom.
    headroom: bool = False
    #: It starts while the ramp spaces admissions.
    ramp: bool = False
    #: It starts before requests queued ahead of it.
    backfilled: bool = False

    @property
    def starts(self) -> bool:
        return self.code == "start"


def limit(limits: Limits, k: float) -> float:
    """The most tokens all runs may hold once a run claiming `k` starts."""
    return limits.tokens + (limits.headroom if k == 1 else 0)


def thin(limits: Limits, free: Optional[int]) -> bool:
    """Whether free memory is unreadable or near its minimum, so the ramp spaces admissions."""
    return free is None or free < limits.min_free + limits.ramp_margin


def is_aged(waiter: Mapping[str, object], limits: Limits, now: float) -> bool:
    """Whether a queued request has waited AGENT_GATE_BACKFILL_SECONDS."""
    return now - number(waiter.get("queued"), now) >= limits.backfill


def _hold(code: str, reason: str) -> Decision:
    return Decision(code, reason)


def queue_decision(ahead: Sequence[Mapping[str, object]], k: int, limits: Limits, now: float,
                   snapshot: Optional[Snapshot] = None) -> Optional[Decision]:
    """Why a request claiming `k` may not pass the requests queued ahead of it; None when it may.

    `snapshot` is None before the runs are read, when only the records judge.
    """
    for waiter in ahead:
        if number(waiter.get("k")) < 1:
            return _hold("malformed", "an earlier request's record names no claim")
    if any(is_aged(waiter, limits, now) for waiter in ahead) and (k != 1 or (snapshot is not None and not (
            snapshot.outside_headroom >= limits.tokens and snapshot.used + 1 <= limit(limits, 1)))):
        return _hold("aged", f"an earlier request has waited over {limits.backfill:g}s for its tokens; nothing "
                             f"passes it except a 1-token run into the headroom while runs outside it hold all "
                             f"{limits.tokens} tokens")
    if snapshot is not None and any(snapshot.used + number(w.get("k")) <= limit(limits, number(w.get("k")))
                                    for w in ahead):
        return _hold("fits-first", "an earlier request that fits goes first")
    return None


def memory_hold(snapshot: Snapshot, limits: Limits, small: bool) -> Optional[Decision]:
    """Why memory has no room for more work beside the live runs; None when it has.

    A small (1-token) addition is held only by measured harm, a larger one by the free-share prediction too.
    """
    if snapshot.pressure is not None and snapshot.pressure >= CRITICAL:
        return _hold("critical", "memory pressure is critical")
    if snapshot.free is None:
        return _hold("unreadable", "memory signal unavailable, so one run at a time")
    if snapshot.swapout is not None and snapshot.swapout > limits.max_swapout:
        return _hold("swap", f"the host swaps out {snapshot.swapout:.1f} MiB/s, over {limits.max_swapout:g}")
    if small:
        return None
    if snapshot.free < limits.min_free:
        return _hold("free", f"{snapshot.free}% memory free, below {limits.min_free:g}%")
    if snapshot.swapout is None:
        return _hold("measuring", "measuring whether the host swaps")
    return None


def fan_out(snapshot: Snapshot, k: int, grow: bool, limits: Limits) -> int:
    """The CPUs a run claiming `k` gets for its runners: see the module docstring."""
    held = memory_hold(snapshot, limits, small=False)
    # A swap rate not measured yet is no evidence of harm.
    if not grow or (held is not None and held.code != "measuring"):
        return k
    return max(k, int(limits.tokens - snapshot.workers))


def decide(snapshot: Snapshot, k: int, limits: Limits, ahead: Sequence[Mapping[str, object]] = (),
           now: float = 0.0, grow: bool = False) -> Decision:
    """Whether a request claiming `k` may start now, `ahead` listing the live requests queued before it.

    With `grow` (a default claim), a request that starts gets the idle CPUs as its fan-out.
    """
    if snapshot.pressure is not None and snapshot.pressure >= CRITICAL:
        return _hold("critical", "memory pressure is critical")
    if ahead:
        held = queue_decision(ahead, k, limits, now, snapshot)
        if held is not None:
            return held
    ramp = False
    if snapshot.used > 0:
        if snapshot.free is None:
            return _hold("unreadable", "memory signal unavailable, so one run at a time")
        if snapshot.used + k > limit(limits, k):
            extra = f" (+{limits.headroom} for 1-token runs)" if k == 1 and limits.headroom else ""
            return _hold("tokens", f"{snapshot.used:g} of {limits.tokens} tokens in use{extra}, {k} needed")
        held = memory_hold(snapshot, limits, small=k == 1)
        if held is not None:
            return held
        ramp = k != 1 and limits.ramp > 0 and thin(limits, snapshot.free)
        if ramp and snapshot.since_admit < limits.ramp:
            return _hold("ramp", "the previous run is still starting")
    return Decision("start", workers=fan_out(snapshot, k, grow, limits),
                    headroom=snapshot.used + k > limits.tokens, ramp=ramp, backfilled=bool(ahead))
