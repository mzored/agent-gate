#!/usr/bin/env python3
"""agent-gate: run a heavy command when this machine has room for it.

Many agents share one Mac. Each gated command declares k CPU tokens, of which
it holds at most half of T (AGENT_GATE_MAX_CLAIM); running commands never hold
more than T tokens together (T = logical CPUs), but for a few tokens of headroom
only 1-token commands may take (AGENT_GATE_SMALL_HEADROOM), and a new
command waits while the kernel reports critical memory pressure or too little
free memory. Waiters form a FIFO queue built on flock(2): each waiter's file
stays locked while it waits, and the kernel drops every lock of a process that
dies, so there is no daemon, lease or registry to repair. Waiting polls (the
head four times a second, the others twice); only the brief registration and
admission locks block. The gate is macOS-only. A later request may start
before earlier ones that wait for tokens alone, when its claim fits now, until
one of them has waited AGENT_GATE_BACKFILL_SECONDS; after that nothing
passes it except a 1-token run into the headroom while runs outside the
headroom hold all T tokens. An aged claim above 1 token cannot start then
anyway, and an aged 1-token request that fits goes first. Every admission is
decided and published under one host lock. While an older gate (one without
that lock) still runs during a rollout, its queue head admits without the lock;
that stays safe because nothing passes an older gate's waiter.

A command whose work runs in Docker containers also declares the memory it needs
in the Docker VM (--docker-gib). It first waits in a queue of its own until the
VM has that much available beyond what earlier Docker runs reserved, then joins
the token queue, so waiting for the VM holds no tokens and delays no other run.

    agent-gate [-c K] [-l LABEL] [--docker-gib G [--docker-frees PROJECT]]
               [--wait SECS | --no-wait] [--timeout SECS] -- CMD...
    agent-gate status [--json]
    agent-gate usage [--hours H] [--match TEXT] [--json]   claimed tokens against measured CPUs
    agent-gate inside [-c K] exit 0 inside a live gated run or while the gate is off (78 if its budget is below K, capped as a claim), 1 outside
    agent-gate off | on      stop or resume admission control host-wide

The command runs in the caller's process group, so a harness or terminal that
stops the group stops the command too. Its environment carries
AGENT_GATE_RUN=<id>; when it exits, tagged leftovers are terminated. The run's
job lock is inherited by the command, so killing the gate alone cannot free
tokens while the command still runs; the next queue head stops such orphans.

A gate started inside a live run never waits: it runs in the parent's budget and
under the parent's fuse, and applies no --timeout of its own.

`agent-gate off` turns admission control off for every gate on the host until
`agent-gate on`: the gate then execs the command as given, as if it were not
there, and `inside` exits 0. A request already queued leaves its queues and
starts the same way within about a second.

Exit codes: the command's own (128+N after signal N); 2 usage error; 75 not
admitted within --wait; 78 bad configuration or unusable state directory; 124
the --timeout fuse stopped the run; 126 or 127 the command could not be
started; 130 Ctrl-C while queued.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from gate_macos import (
    GRACE_SECONDS, RUN_ENV, Processes, ancestors, host_busy, host_cpu_ticks,
    lock_holders, memory_signal, own_start, same_process, stop_processes,
    sweep_run, tagged_pids, yield_to_the_owner,
)

NOT_ADMITTED_EXIT = 75
CONFIG_EXIT = 78
TIMEOUT_EXIT = 124

DIR_ENV = "AGENT_GATE_DIR"
CPUS_ENV = "AGENT_GATE_CPUS"
TOKENS_ENV = "AGENT_GATE_TOKENS"
MAX_CLAIM_ENV = "AGENT_GATE_MAX_CLAIM"
MIN_FREE_ENV = "AGENT_GATE_MIN_FREE_PCT"
RAMP_ENV = "AGENT_GATE_RAMP_SECONDS"
#: The ramp spaces admissions only while free memory is within this many points of the minimum.
RAMP_MARGIN_ENV = "AGENT_GATE_RAMP_MARGIN_PCT"
#: How long a queued request may be passed by later ones before only 1-token runs into the headroom
#: pass it, while runs outside the headroom hold all T tokens.
BACKFILL_ENV = "AGENT_GATE_BACKFILL_SECONDS"
#: Tokens beyond T that only 1-token runs may hold.
HEADROOM_ENV = "AGENT_GATE_SMALL_HEADROOM"
#: While other runs run, no new run starts while the host swaps out more than this many MiB/s.
MAX_SWAPOUT_ENV = "AGENT_GATE_MAX_SWAPOUT_MBPS"
#: Images the Docker VM probe may run, first present first; when none is present, the
#: first is pulled.
PROBE_IMAGES_ENV = "AGENT_GATE_DOCKER_PROBE_IMAGES"
#: How long a Docker claim must exceed what the VM could offer before it is refused.
CEILING_ENV = "AGENT_GATE_DOCKER_CEILING_SECONDS"

#: Shimmed runs without -c used p90 1.7 and p95 2.7 CPUs over 72 h (2026-10-19), and
#: their runners start this many workers, each with its own memory.
DEFAULT_CPUS = 2
DEFAULT_TIMEOUT = 3600.0
#: The free share read 20-24% on a calm host and 11-12% while it swapped (2026-10-21).
DEFAULT_MIN_FREE_PCT = 15.0
DEFAULT_RAMP_SECONDS = 5.0
DEFAULT_RAMP_MARGIN_PCT = 10.0
DEFAULT_BACKFILL_SECONDS = 60.0
DEFAULT_MAX_SWAPOUT_MBPS = 10.0
#: A waiter record of this protocol or later admits only under the admit lock, so
#: later requests may pass it. An older gate's record has none: it admits at the
#: head without that lock, so nothing may pass it while it waits.
PROTOCOL = 2
#: Memory a build, a pull or a stopping container takes shows in no running container,
#: so only a claim the VM falls short of at every survey this long is a configuration error.
DEFAULT_CEILING_SECONDS = 300.0
POLL_SECONDS = 0.25
#: How often a waiter checks the request ahead of it, and collects dead gates' runs.
WAITER_POLL_SECONDS = 0.5
SWEEP_SECONDS = 30.0
PROGRESS_SECONDS = 60.0
SAMPLE_SECONDS = 5.0
#: After a forwarded INT/TERM/HUP, how long the command may take to wind down.
CANCEL_SECONDS = 10.0
#: The longest pause between attempts to stop what one dead gate's run left behind.
ORPHAN_RETRY_SECONDS = 60.0
#: A staging file untouched this long was left by a dead creator.
STAGING_SECONDS = 60.0
#: A request that may not wait still gets this long to register while another registers.
REGISTER_GRACE_SECONDS = 1.0
#: How often the Docker queue's head reads the VM's memory again.
DOCKER_POLL_SECONDS = 5.0
#: One docker call; a wedged daemon must not hold the queue longer than this.
DOCKER_SECONDS = 30.0
#: What one reading of the VM gets in all, however little the wait has left: a
#: `docker stats` alone takes about two seconds.
DOCKER_MIN_SECONDS = 10.0
#: How long `status` shows the Docker queue head's last reading of the VM.
VM_READING_SECONDS = 600.0
PROBE_IMAGES = ("alpine:latest", "busybox:latest", "redis:7-alpine", "redis:7", "postgres:16-alpine",
                "postgres:15", "python:3.12-slim", "python:3.11-slim")
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
GIB = 2**30
LOG_LIMIT_BYTES = 20 * 2**20
#: By absolute path: callers' PATH often leaves out /usr/sbin.
CRITICAL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}

#: Worker limits the runners read, so a run's fan-out equals its tokens.
#: Vitest 4 reads VITEST_MAX_WORKERS; Vitest 3 reads VITEST_MAX_THREADS/FORKS;
#: pytest-xdist reads PYTEST_XDIST_AUTO_NUM_WORKERS for `-n auto`.
RUNNER_ENV = ("VITEST_MAX_WORKERS", "VITEST_MAX_THREADS", "VITEST_MAX_FORKS", "PYTEST_XDIST_AUTO_NUM_WORKERS")

WAITER_RE = re.compile(r"^(\d{12})-([0-9a-f]{16})\.lock$")
JOB_RE = re.compile(r"^([0-9a-f]{16})\.lock$")
STAGING_RE = re.compile(r"^\..+\.tmp$")
#: When to try again to stop a dead gate's run; a name older gates' scans pass over.
RETRY_RE = re.compile(r"^([0-9a-f]{16})\.retry$")


class ConfigError(Exception):
    """The request or the state directory cannot be used; nothing ran."""


class WaitExpired(Exception):
    """--wait elapsed before admission; nothing ran."""


class SwitchedOff(Exception):
    """Admission control was switched off while the request waited; it leaves the queues and starts at once."""


def check_off(settings: "Settings") -> None:
    if settings.off.exists():
        raise SwitchedOff()


def say(message: str) -> None:
    print(f"agent-gate: {message}", file=sys.stderr, flush=True)


def unusable(error: Exception) -> int:
    """Report a request or state directory that cannot be used, and the exit code that means so."""
    say(f"unusable: {error}")
    return CONFIG_EXIT


# ------------------------------------------------------------------ kernel


def try_lock(fd: int, mode: int) -> bool:
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        raise


def read_json_fd(fd: int) -> Dict[str, object]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        value = json.loads(b"".join(chunks) or b"{}")
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite(text: object, name: str) -> float:
    """A finite, non-negative number from the command line or the environment."""
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ConfigError(f"{name} {text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} {text!r} must be a finite number of at least 0")
    return value


# ------------------------------------------------------------------ state


class Settings:
    def __init__(self, environ: Mapping[str, str]):
        self.environ = environ
        self.tokens = self._int(TOKENS_ENV, os.cpu_count() or 1)
        # The most one run holds, so that no single claim drains the machine.
        self.max_claim = min(self.tokens, self._int(MAX_CLAIM_ENV, max(1, self.tokens // 2)))
        self.min_free = self._float(MIN_FREE_ENV, DEFAULT_MIN_FREE_PCT)
        self.ramp = self._float(RAMP_ENV, DEFAULT_RAMP_SECONDS)
        self.ramp_margin = self._float(RAMP_MARGIN_ENV, DEFAULT_RAMP_MARGIN_PCT)
        self.max_swapout = self._float(MAX_SWAPOUT_ENV, DEFAULT_MAX_SWAPOUT_MBPS)
        self.backfill = self._float(BACKFILL_ENV, DEFAULT_BACKFILL_SECONDS)
        # Focused 1-token checks use most of what they claim while large claims use less,
        # so they may run beside a full pool.
        self.headroom = self._int(HEADROOM_ENV, max(1, self.tokens // 5), minimum=0)
        self.docker_ceiling = self._float(CEILING_ENV, DEFAULT_CEILING_SECONDS)
        root = environ.get(DIR_ENV) or f"/tmp/agent-gate-{os.getuid()}"
        self.root = Path(root)
        self.waiters = self.root / "waiters"
        self.jobs = self.root / "jobs"
        self.admit_lock = self.root / "admit.lock"
        self.last_admit = self.root / "last-admit"
        self.events = self.root / "events.jsonl"
        # The Docker VM's memory has a queue and reservations of its own.
        self.docker = self.root / "docker"
        self.docker_waiters = self.docker / "waiters"
        self.docker_jobs = self.docker / "jobs"
        self.docker_vm = self.docker / "vm.json"
        #: Present while admission control is off host-wide (`agent-gate off`).
        self.off = self.root / "off"
        images = (environ.get(PROBE_IMAGES_ENV) or "").replace(",", " ").split() or PROBE_IMAGES
        # `docker image ls` names an image by repository and tag.
        self.probe_images = tuple(name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest" for name in images)

    def _int(self, name: str, default: int, minimum: int = 1) -> int:
        text = self.environ.get(name)
        if not text:
            return default
        try:
            value = int(text)
        except ValueError:
            raise ConfigError(f"{name}={text!r} is not an integer") from None
        if value < minimum:
            raise ConfigError(f"{name} must be at least {minimum}")
        return value

    def limit(self, k: float) -> int:
        """The most tokens all runs may hold once a run claiming `k` starts."""
        return self.tokens + (self.headroom if k == 1 else 0)

    def ramps(self, free: Optional[int]) -> bool:
        """Whether admissions are spaced by the ramp: free memory is unreadable or near its minimum."""
        return free is None or free < self.min_free + self.ramp_margin

    def _float(self, name: str, default: float) -> float:
        text = self.environ.get(name)
        if not text:
            return default
        return finite(text, name)

    def ensure(self) -> None:
        for path in (self.root, self.waiters, self.jobs, self.docker, self.docker_waiters, self.docker_jobs):
            with contextlib.suppress(FileExistsError):
                os.mkdir(path, 0o700)
            info = os.lstat(path)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ConfigError(f"{path} must be a directory owned by uid {os.getuid()} with mode 0700")


def load_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    """The settings of `environ`, a copy of it; the process environment by default. ConfigError when they are bad."""
    return Settings(dict(os.environ if environ is None else environ))


@contextlib.contextmanager
def exclusive(path: Path, deadline: Optional[float] = None, what: str = "queue registration") -> Iterator[int]:
    """Hold `path` exclusively; with a deadline, give up when it passes."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if deadline is None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            while not try_lock(fd, fcntl.LOCK_EX):
                if time.monotonic() >= deadline:
                    raise WaitExpired(f"another request held the {what} lock")
                time.sleep(0.02)
        yield fd
    finally:
        os.close(fd)


def publish_locked(directory: Path, name: str, record: Mapping[str, object]) -> int:
    """Create `directory/name` already holding LOCK_EX, so no prober sees it unlocked."""
    staging = directory / f".{name}.tmp"
    fd = os.open(staging, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, json.dumps(record, sort_keys=True).encode())
        os.rename(staging, directory / name)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(staging)
        raise
    return fd


def _collect_staging(path: Path) -> None:
    """Remove a staging file its dead creator left behind."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        if try_lock(fd, fcntl.LOCK_SH) and time.time() - os.fstat(fd).st_mtime > STAGING_SECONDS:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
    finally:
        os.close(fd)


def _listdir(path: Path) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []


def _lock_files(directory: Path, pattern: "re.Pattern[str]", collect: bool) -> Iterator[Tuple["re.Match[str]", Path]]:
    """The files in `directory` whose names match `pattern`, in name order, with each name's match.

    With `collect`, staging files their dead creators left behind are removed on the way.
    """
    for name in _listdir(directory):
        match = pattern.match(name)
        if match:
            yield match, directory / name
        elif collect and STAGING_RE.match(name):
            _collect_staging(directory / name)


@contextlib.contextmanager
def _probed(path: Path) -> Iterator[Tuple[int, bool]]:
    """`path` open read-only, and whether a live process holds its lock.

    When none does, this process holds it shared until the block ends, so its
    file can be removed before anyone else takes it. FileNotFoundError when the
    file is gone.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        yield fd, not try_lock(fd, fcntl.LOCK_SH)
    finally:
        os.close(fd)


def live_waiters(settings: Settings, before: Optional[int] = None, collect: bool = True,
                 directory: Optional[Path] = None) -> List[Dict[str, object]]:
    """Queued requests in ticket order (the token queue's unless `directory` names another); with `collect`,
    files of dead waiters are removed and each one's request is logged as abandoned."""
    directory = settings.waiters if directory is None else directory
    result: List[Dict[str, object]] = []
    for match, path in _lock_files(directory, WAITER_RE, collect):
        ticket = int(match.group(1))
        if before is not None and ticket >= before:
            continue
        try:
            with _probed(path) as (fd, held):
                if not held:
                    if collect:
                        record = read_json_fd(fd)
                        os.unlink(path)  # whoever unlinks a dead waiter's file logs it, once
                        log_abandoned(settings, directory, record)
                    continue
                record = read_json_fd(fd)
        except FileNotFoundError:
            continue
        record.update(ticket=ticket, path=str(path))
        result.append(record)
    return result


def log_abandoned(settings: Settings, directory: Path, record: Mapping[str, object]) -> None:
    """Log a request whose gate died before its turn, killed by a caller's timeout, say.

    A gate that leaves on its own (wait limit, Ctrl-C) removes its file and logs
    not-admitted; only the file of a dead waiter is left to collect, and
    whoever unlinks it logs it, once. The waiter right behind wakes when the
    lock goes, so the wait is exact but for the last waiter in line, whose file
    waits for the next request and so overstates it.
    """
    queued = number(record.get("queued"), 0)
    log_event(settings, {
        "event": "abandoned",
        "queue": "docker" if directory == settings.docker_waiters else "tokens",
        **{key: record.get(key) for key in ("id", "label", "cwd", "k", "docker_gib", "pid")},
        "waited_at_most_s": round(time.time() - queued, 3) if queued else None,
    })


def scan_jobs(settings: Settings, sweep: bool = False) -> List[Dict[str, object]]:
    """Runs that may still use their tokens.

    A run counts while its job lock is held, and, once its gate died, until a
    queue head has verified that nothing of it is left. With `sweep` (the queue
    head only), runs whose gate died are stopped and their files removed.
    """
    live: List[Dict[str, object]] = []
    for _, path in _lock_files(settings.jobs, JOB_RE, sweep):
        try:
            with _probed(path) as (fd, held):
                record = read_json_fd(fd)
        except FileNotFoundError:
            continue
        run_id = path.name[: -len(".lock")]
        gate_alive = same_process(record.get("pid"), record.get("start"))
        record.update(id=run_id, gate_alive=gate_alive, held=held, path=str(path))
        if gate_alive and not held:
            continue  # its gate is releasing it
        if not gate_alive and sweep and _collect_orphan(settings, record, path):
            continue
        live.append(record)
    if sweep:
        # Retry state whose run is gone, as when an older gate removed it.
        for match, retry in _lock_files(settings.jobs, RETRY_RE, False):
            if not (settings.jobs / f"{match.group(1)}.lock").exists():
                with contextlib.suppress(OSError):
                    os.unlink(retry)
    return live


def _collect_orphan(settings: Settings, record: Dict[str, object], path: Path) -> bool:
    """Stop what a dead gate's run left behind; True once nothing of it remains and its files are gone.

    While survivors remain, or cannot be seen, the run keeps its tokens: a
    sandboxed head that may not signal them leaves them to a head outside.
    One gate tries at a time, and the pause after a failed attempt holds for
    every gate: the run's `.retry` file keeps it, locked while a gate tries.
    """
    run_id = str(record["id"])
    retry = settings.jobs / f"{run_id}.retry"
    try:
        fd: Optional[int] = os.open(retry, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        fd = None  # nowhere to keep the pause, as on a full disk: this attempt goes unpaced
    try:
        failures = 0
        if fd is not None:
            if not try_lock(fd, fcntl.LOCK_EX):
                return False  # another gate is trying now
            state = read_json_fd(fd)
            failures = int(number(state.get("failures")))
            now = time.time()
            # A pause longer than the longest was set by a clock since set back: it has run out.
            if now < number(state.get("next")) <= now + ORPHAN_RETRY_SECONDS:
                return False
        since = number(record.get("admitted"))

        def owned() -> Optional[Processes]:
            tagged, holders = tagged_pids(run_id), lock_holders(path, since)
            return None if tagged is None or holders is None else {**tagged, **holders}

        found = owned()
        stopped = stop_processes(found, owned) if found else 0
        remaining = owned() if found else found
        gone = remaining == {} and _remove_unlocked(path)
        if gone:
            with contextlib.suppress(OSError):
                os.unlink(retry)
        elif fd is not None:
            # A holder still exiting is gone within a second; survivors this head may
            # not signal, or an unreadable process table, are retried ever more rarely.
            delay = min(ORPHAN_RETRY_SECONDS, 2.0 ** min(failures, 6))
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
                os.pwrite(fd, json.dumps({"failures": failures + 1, "next": round(time.time() + delay, 3)}).encode(), 0)
    finally:
        if fd is not None:
            os.close(fd)
    log_event(settings, {
        "event": "orphan-stopped" if gone else "orphan-survives",
        "id": run_id,
        "label": record.get("label"),
        "pids": stopped,
        "remaining": None if remaining is None else len(remaining),
    })
    return gone


def _remove_unlocked(path: Path) -> bool:
    """Remove `path` unless a process still holds its lock; True when it is gone."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return True
    try:
        if not try_lock(fd, fcntl.LOCK_EX):
            return False
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        return True
    finally:
        os.close(fd)


def log_event(settings: Settings, event: Mapping[str, object]) -> None:
    line = json.dumps(dict(event, ts=round(time.time(), 3), v=1), sort_keys=True) + "\n"
    try:
        fd = os.open(settings.events, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size > LOG_LIMIT_BYTES:
                _rotate(settings.events, fd)
            # The line that found the log full ends it, wherever it was moved.
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def _rotate(path: Path, fd: int) -> None:
    """Move the full log at `path`, open as `fd`, to `path`.1 unless another gate is moving it.

    It moves under the log's own lock, and only while `path` still names the
    full log, so gates that find it full at once move it once and never move a
    fresh log over the full one. An older gate moves it without the lock.
    """
    if not try_lock(fd, fcntl.LOCK_EX):
        return
    try:
        current = os.stat(path)
    except FileNotFoundError:
        return
    opened = os.fstat(fd)
    if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
        os.replace(path, path.with_name(path.name + ".1"))


# ------------------------------------------------------------------ decisions


class Snapshot:
    def __init__(self, settings: Settings, sweep: bool):
        self.jobs = scan_jobs(settings, sweep=sweep)
        self.pressure, self.free, self.swapout = memory_signal(settings.environ)
        try:
            self.since_admit = time.time() - settings.last_admit.stat().st_mtime
        except OSError:
            self.since_admit = float("inf")

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
        """Tokens of `used` held by runs not admitted into the headroom; an older gate's runs count here."""
        return self.used - self.in_headroom

    def memory_text(self) -> str:
        if self.free is None:
            return "memory signal unavailable"
        level = "pressure unreadable" if self.pressure is None else PRESSURE_NAMES.get(self.pressure, self.pressure)
        swap = "" if self.swapout is None else f", swapping out {self.swapout:.1f} MiB/s"
        return f"memory {level}, {self.free}% free{swap}"


def is_aged(waiter: Mapping[str, object], settings: Settings, now: float) -> bool:
    """Whether a queued request has waited AGENT_GATE_BACKFILL_SECONDS."""
    return now - number(waiter.get("queued"), now) >= settings.backfill


def passing_refusal(ahead: Sequence[Mapping[str, object]], k: int, settings: Settings, now: float,
                    snapshot: Optional[Snapshot] = None) -> Optional[str]:
    """Why a request claiming `k` may not pass the requests queued ahead of it.

    An older gate's request admits at the head without the admit lock, so
    nothing passes it. Nothing passes a request that has waited
    AGENT_GATE_BACKFILL_SECONDS except a 1-token run into the headroom while
    runs outside the headroom hold all T tokens by themselves. An aged claim
    above 1 token needs room within T, so it cannot start then anyway, and
    once one of those runs ends the passing stops, so it waits only for runs
    that already exist. An aged 1-token request fits whenever the passer does,
    so the rule that a request ahead that fits goes first protects it.
    `snapshot` is None before the runs are read, when only the records judge.
    """
    aged = False
    for waiter in ahead:
        if number(waiter.get("protocol")) < PROTOCOL or number(waiter.get("k")) < 1:
            return "an earlier request admits only at the head of the queue (an older gate's)"
        aged = aged or is_aged(waiter, settings, now)
    if aged and (k != 1 or (snapshot is not None and not (
            snapshot.outside_headroom >= settings.tokens and snapshot.used + 1 <= settings.limit(1)))):
        return (f"an earlier request has waited over {settings.backfill:g}s for its tokens; nothing passes it "
                f"except a 1-token run into the headroom while runs outside it hold all {settings.tokens} tokens")
    return None


def refusal(snapshot: Snapshot, k: int, settings: Settings, ahead: Sequence[Mapping[str, object]] = (),
            now: Optional[float] = None) -> Optional[str]:
    """Why a request cannot start now; None when it can.

    `ahead` lists the live requests queued before it. With none, it is the
    head and starts once the host has room. A later request starts first only
    while each one ahead waits for tokens alone: memory and the ramp hold every
    request alike, so one they hold is never passed, and one that fits goes first.
    """
    if snapshot.pressure is not None and snapshot.pressure >= CRITICAL:
        return "memory pressure is critical"
    if ahead:
        reason = passing_refusal(ahead, k, settings, time.time() if now is None else now, snapshot)
        if reason is not None:
            return reason
        if any(snapshot.used + number(waiter.get("k")) <= settings.limit(number(waiter.get("k"))) for waiter in ahead):
            return "an earlier request that fits goes first"
    if snapshot.used <= 0:
        return None
    if snapshot.free is None:
        return "memory signal unavailable, so one run at a time"
    if snapshot.used + k > settings.limit(k):
        extra = f" (+{settings.headroom} for 1-token runs)" if k == 1 and settings.headroom else ""
        return f"{snapshot.used:g} of {settings.tokens} tokens in use{extra}, {k} needed"
    if snapshot.free < settings.min_free:
        return f"{snapshot.free}% memory free, below {settings.min_free:g}%"
    if snapshot.swapout is None:
        return "measuring whether the host swaps"
    if snapshot.swapout > settings.max_swapout:
        return f"the host swaps out {snapshot.swapout:.1f} MiB/s, over {settings.max_swapout:g}"
    if settings.ramps(snapshot.free) and snapshot.since_admit < settings.ramp:
        return "the previous run is still starting"
    return None


def enclosing_run(settings: Settings, environ: Mapping[str, str]) -> Optional[Dict[str, object]]:
    """The live run this process belongs to, proven by ancestry; inherited variables alone prove nothing.

    The result carries the run's budget `k` and its `id`, the run tag its commands keep.
    """
    chain = ancestors()
    # `gate_alive` is `same_process` on the record, so an ancestor's pid alone completes the identity.
    parent = next(
        (job for job in scan_jobs(settings) if job["gate_alive"] and int(number(job.get("pid"), -1)) in chain),
        None,
    )
    # A nested gate in between exported a smaller budget; a budget only ever shrinks.
    inherited = math.floor(number(environ.get(CPUS_ENV)))
    if parent is not None and inherited >= 1:
        parent["k"] = min(int(number(parent.get("k"))), inherited)
    return parent


# ------------------------------------------------------------------ Docker VM
#
# Containers run in Docker Desktop's VM, whose memory the host's signal does not
# show and which every project's stacks share. A run that declares --docker-gib
# reserves that much of it: the Docker queue's head starts when what the VM has
# available, less what earlier Docker runs reserved, covers its claim.


class DockerUnreadable(Exception):
    """Docker did not answer, or its answer made no sense."""


#: The time one docker call may take; a request's wait limit shortens it.
Limit = Callable[[], float]


def docker(arguments: Sequence[str], timeout: float = DOCKER_SECONDS) -> str:
    if timeout <= 0:
        raise DockerUnreadable(f"docker {arguments[0]}: no time left to ask")
    try:
        result = subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise DockerUnreadable(f"docker {arguments[0]}: {error}") from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise DockerUnreadable(f"docker {arguments[0]} exited {result.returncode}: {detail[-1] if detail else ''}")
    return result.stdout


def docker_memory(settings: Settings, limit: Limit = lambda: DOCKER_SECONDS) -> Tuple[float, float]:
    """GiB available and in total where containers run, read from /proc/meminfo in a short-lived container.

    Docker Desktop shows containers its VM's meminfo; a Linux host shows them
    its own, which is where they run there. The container runs an image already
    present, so the probe pulls nothing while one of them exists.
    """
    present = set(docker(["image", "ls", "--format", "{{.Repository}}:{{.Tag}}"], limit()).split())
    image = next((name for name in settings.probe_images if name in present), None)
    meminfo = docker([
        "run", "--rm", f"--pull={'never' if image else 'missing'}", "--network=none", "--log-driver=none",
        "--entrypoint", "cat", image or settings.probe_images[0], "/proc/meminfo",
    ], limit())
    fields = dict(re.findall(r"^(MemAvailable|MemTotal):\s+(\d+) kB\s*$", meminfo, re.MULTILINE))
    if len(fields) != 2:
        raise DockerUnreadable("no MemAvailable and MemTotal in the container's /proc/meminfo")
    return int(fields["MemAvailable"]) * 1024 / GIB, int(fields["MemTotal"]) * 1024 / GIB


_UNITS = {"b": 1, "kib": 2**10, "kb": 10**3, "mib": 2**20, "mb": 10**6, "gib": 2**30, "gb": 10**9}


def containers(limit: Limit = lambda: DOCKER_SECONDS) -> List[Tuple[str, str, float]]:
    """Running containers: their compose project ("" outside compose), name and the GiB each uses."""
    owners: Dict[str, Tuple[str, str]] = {}
    listing = docker(["ps", "--format", '{{.ID}}\t{{.Label "' + COMPOSE_PROJECT_LABEL + '"}}\t{{.Names}}'], limit())
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            owners[parts[0]] = (parts[1], parts[2])
    if not owners:
        return []
    # Every running container, not the ones above: one that exits in between must not fail the call.
    found = []
    for line in docker(["stats", "--no-stream", "--format", "{{.ID}}\t{{.MemUsage}}"], limit()).splitlines():
        parts = line.split("\t")
        match = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", parts[-1].split("/", 1)[0])
        if len(parts) == 2 and parts[0] in owners and match and match.group(2).lower() in _UNITS:
            found.append((*owners[parts[0]], float(match.group(1)) * _UNITS[match.group(2).lower()] / GIB))
    return found


def by_project(found: Iterable[Tuple[str, str, float]]) -> List[Tuple[str, float]]:
    """GiB each compose project uses, largest first; a container outside compose counts under its name."""
    totals: Dict[str, float] = {}
    for project, name, gib in found:
        totals[project or name] = totals.get(project or name, 0.0) + gib
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def docker_holders(settings: Settings, collect: bool = False) -> List[Dict[str, object]]:
    """Docker VM reservations still held; with `collect`, released ones' files are removed.

    A reservation's lock is inherited by its command, so it holds while the
    command runs, even after its gate died.
    """
    found: List[Dict[str, object]] = []
    for _, path in _lock_files(settings.docker_jobs, JOB_RE, collect):
        try:
            with _probed(path) as (fd, held):
                if not held:
                    if collect:
                        os.unlink(path)
                    continue
                record = read_json_fd(fd)
        except FileNotFoundError:
            continue
        record.update(id=path.name[: -len(".lock")], path=str(path))
        found.append(record)
    return found


def reserved_gib(holders: Iterable[Mapping[str, object]]) -> float:
    """GiB of the Docker VM the reservations in `holders` hold."""
    return sum(number(holder.get("docker_gib")) for holder in holders)


def save_vm(settings: Settings, vm: Mapping[str, object]) -> None:
    """Keep the head's last reading of the VM for `status`; the queue has one head at a time."""
    staging = settings.docker / ".vm.json.tmp"
    with contextlib.suppress(OSError):
        staging.write_text(json.dumps(dict(vm, at=round(time.time(), 3)), sort_keys=True))
        os.replace(staging, settings.docker_vm)


def docker_refusal(settings: Settings, request: "Request", holders: Sequence[Mapping[str, object]],
                   vm: Dict[str, object], limit: Limit, survey: bool) -> Optional[str]:
    """Why the Docker queue's head cannot start now; None when it can.

    Fills `vm` with what it read of the VM, for the admitted record and for
    `status`. With `survey`, or a project to credit, a refused head lists the
    running containers. It names the largest users in `vm["users"]`, and puts in
    `vm["ceiling_gib"]` what the VM could offer with every container
    stopped and every Docker run ended.
    """
    reserved = reserved_gib(holders)
    vm["reserved_gib"] = round(reserved, 3)
    try:
        available, total = docker_memory(settings, limit)
    except DockerUnreadable as error:
        vm["error"] = str(error)
        if not holders:
            return None
        return f"the Docker VM's memory is unreadable ({error}), so one Docker run at a time"
    vm.update(available_gib=round(available, 3), total_gib=round(total, 3))
    if request.docker_gib > total:
        raise ConfigError(f"--docker-gib {request.docker_gib:g} exceeds the Docker VM's {total:.2f} GiB")
    if available - reserved >= request.docker_gib:
        return None
    in_use: Optional[List[Tuple[str, str, float]]] = None
    if survey or request.docker_frees:
        with contextlib.suppress(DockerUnreadable):
            in_use = containers(limit)
    freed = 0.0
    if request.docker_frees and in_use is not None:
        # By compose label only: a container merely named like the project is not stopped by it.
        freed = sum(gib for project, _, gib in in_use if project == request.docker_frees)
        vm["freed_gib"] = round(freed, 3)
        if available + freed - reserved >= request.docker_gib:
            return None
    text = f"the Docker VM has {available:.2f} GiB available"
    if freed:
        text += f" and {request.docker_frees} frees {freed:.2f} GiB"
    if reserved:
        text += f", {reserved:.2f} GiB of it reserved by {len(holders)} earlier run(s)"
    text += f"; {request.docker_gib:g} GiB needed"
    if request.docker_frees and in_use is None:
        text += f" (container usage unreadable, so nothing credited for {request.docker_frees})"
    if in_use is not None:
        ceiling = available + reserved + sum(gib for _, _, gib in in_use)
        vm["ceiling_gib"] = round(ceiling, 3)
        if request.docker_gib > ceiling:
            text += f"; with every container stopped it could offer {ceiling:.2f} GiB now"
        shown = ", ".join(f"{name} {gib:.2f} GiB" for name, gib in by_project(in_use)[:5])
        vm["users"] = shown or "no running containers"
    return text


# ------------------------------------------------------------------ waiting


class Progress:
    """Tells a waiting caller where it stands: at once, then every PROGRESS_SECONDS.

    With `docker_gib`, it reports on the Docker queue instead of the token queue.
    """

    def __init__(self, settings: Settings, label: str, k: int, docker_gib: float = 0.0):
        self.settings = settings
        self.label = label
        self.k = k
        self.docker_gib = docker_gib
        self.ticket: Optional[int] = None
        self.reason = ""
        self.started = time.monotonic()
        self.announced = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def describe(self) -> str:
        directory = self.settings.docker_waiters if self.docker_gib else self.settings.waiters
        ahead = live_waiters(self.settings, self.ticket, False, directory) if self.ticket else []
        parts = [f"position {len(ahead) + 1}"]
        if self.docker_gib:
            holders = docker_holders(self.settings)
            if holders:
                shown = ", ".join(f"{h.get('label') or h['id']} ({number(h.get('docker_gib')):g} GiB)" for h in holders[:4])
                parts.append(f"reserved in the Docker VM: {shown}")
        else:
            running = scan_jobs(self.settings)
            if running:
                shown = ", ".join(f"{job.get('label') or job['id']} (k={job.get('k')})" for job in running[:4])
                parts.append(f"running: {shown}")
        if self.reason:
            parts.append(f"waiting because {self.reason}")
        return "; ".join(parts)

    def announce(self) -> None:
        if self.announced:
            return
        self.announced = True
        claim = f"for {self.docker_gib:g} GiB of Docker VM memory" if self.docker_gib else f"(k={self.k})"
        say(
            f"queued {self.label} {claim}: {self.describe()}. Waiting is not a failure: "
            "keep this command running (long tool timeout or background)."
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(PROGRESS_SECONDS):
            with contextlib.suppress(Exception):
                say(f"still waiting after {time.monotonic() - self.started:.0f}s: {self.describe()}")

    def stop(self) -> None:
        self._stop.set()


def block_shared(path: str, deadline: Optional[float], idle: Callable[[], object],
                 poll: Callable[[], object] = lambda: None) -> None:
    """Wait until whoever holds `path` exclusively lets go (or dies).

    It polls: it tries the lock every WAITER_POLL_SECONDS and does not block in the kernel.
    Calls `poll` on every check, and `idle` every SWEEP_SECONDS.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        next_idle = time.monotonic() + SWEEP_SECONDS
        while not try_lock(fd, fcntl.LOCK_SH):
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise WaitExpired("still queued behind earlier requests")
            poll()
            if now >= next_idle:
                idle()
                next_idle = time.monotonic() + SWEEP_SECONDS
            time.sleep(WAITER_POLL_SECONDS)
    finally:
        os.close(fd)


class Request:
    def __init__(self, k: int, label: str, argv: Sequence[str], wait: Optional[float], timeout: float,
                 asked: Optional[int] = None, docker_gib: float = 0.0, docker_frees: str = ""):
        self.k = k
        self.asked = k if asked is None else asked
        self.label = label
        self.argv = list(argv)
        self.timeout = timeout
        self.docker_gib = docker_gib
        self.docker_frees = docker_frees
        self.id = secrets.token_hex(8)
        self.started = time.monotonic()
        # One limit covers both queues.
        self.deadline = None if wait is None else self.started + wait
        #: The held Docker VM reservation, once admitted to it.
        self.docker_fd: Optional[int] = None
        self.docker_waited = 0.0


def with_grace(deadline: Optional[float]) -> Optional[float]:
    """A wait limit extended to leave a request that may not wait a moment to take a lock."""
    return None if deadline is None else max(deadline, time.monotonic() + REGISTER_GRACE_SECONDS)


def job_record(request: Request, tokens: bool = True, queued: Optional[float] = None) -> Dict[str, object]:
    """What every queue and job file of a request holds: who asks, for what, since when.

    The Docker queue's files hold no tokens. `queued` is the time the request began to wait; by default, now.
    """
    record: Dict[str, object] = {
        "id": request.id,
        "pid": os.getpid(),
        "start": own_start(),
        "label": request.label,
        "cwd": os.getcwd(),
        "docker_gib": request.docker_gib,
        "queued": round(time.time() if queued is None else queued, 3),
    }
    if tokens:
        record.update(k=request.k, asked=request.asked)
    return record


def register(settings: Settings, queue: Path, request: Request, record: Mapping[str, object]) -> Tuple[int, int, Path]:
    """Take the next ticket of the queue in `queue` and publish our waiter file there.

    Returns the waiter's held lock, its ticket and its path.
    """
    waiters = queue / "waiters"
    # A request that may not wait still gets a moment to register, in each queue it joins.
    with exclusive(queue / "queue.lock", with_grace(request.deadline)):
        try:
            last = int((queue / "ticket").read_text().strip() or 0)
        except (OSError, ValueError):
            last = 0
        existing = [int(m.group(1)) for m in map(WAITER_RE.match, _listdir(waiters)) if m]
        ticket = max([last, *existing]) + 1
        staging = queue / ".ticket.tmp"
        staging.write_text(str(ticket))
        os.replace(staging, queue / "ticket")
        name = f"{ticket:012d}-{request.id}.lock"
        waiter = publish_locked(waiters, name, dict(record, ticket=ticket))
    return waiter, ticket, waiters / name


@contextlib.contextmanager
def queued(settings: Settings, queue: Path, request: Request, record: Mapping[str, object],
           progress: Progress) -> Iterator[int]:
    """Hold a place in the queue in `queue` while the block runs; yields its ticket.

    However the block ends, the request leaves that queue.
    """
    waiter, ticket, waiter_path = register(settings, queue, request, record)
    progress.ticket = ticket
    try:
        yield ticket
    finally:
        progress.stop()
        # Unlinked while still locked, so no collector takes it for a dead waiter's.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(waiter_path)
        os.close(waiter)


def wait_for_turn(settings: Settings, waiters: Path, ticket: int, deadline: Optional[float], progress: Progress) -> None:
    """Wait until no live request with an earlier ticket remains in `waiters`: the Docker queue's strict order."""
    while True:
        check_off(settings)
        ahead = live_waiters(settings, before=ticket, directory=waiters)
        if not ahead:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise WaitExpired(f"{len(ahead)} earlier request(s) still queued")
        progress.announce()
        # A head that may not signal a dead gate's survivors (a sandboxed one)
        # would wait on them forever, so every waiter collects them too.
        block_shared(str(ahead[-1]["path"]), deadline, lambda: scan_jobs(settings, sweep=True),
                     lambda: check_off(settings))


def decide(settings: Settings, k: int, ticket: int, sweep: bool) -> Tuple[Optional[str], int, Optional[Snapshot]]:
    """Why the request holding `ticket` cannot start now (None when it can), how many wait ahead, and the snapshot.

    The queue is read before the runs: a request leaves the queue only after
    publishing its run, so none is missed between the two readings. With
    `sweep`, a head collects dead gates' runs first.
    """
    ahead = live_waiters(settings, before=ticket)
    reason = passing_refusal(ahead, k, settings, time.time())
    if reason is not None:
        return reason, len(ahead), None
    snapshot = Snapshot(settings, sweep=sweep and not ahead)
    return refusal(snapshot, k, settings, ahead), len(ahead), snapshot


def admit(settings: Settings, request: Request, progress: Progress) -> Tuple[int, Dict[str, object], float]:
    """Queue, then start once the host has room and no earlier request must go first.

    Several requests may start while one waits for tokens, so each decides and
    publishes its run under the admit lock. Returns the held job lock, the job
    record and the time waited. Raises SwitchedOff, out of the queue, once
    admission control is switched off.
    """
    deadline = request.deadline
    record = dict(job_record(request), protocol=PROTOCOL)
    with queued(settings, settings.root, request, record, progress) as ticket:
        at_head: Optional[float] = None
        # The log keeps why the run was held back.
        held_back: Optional[str] = None
        next_sweep = time.monotonic() + SWEEP_SECONDS
        while True:
            check_off(settings)
            reason, ahead, snapshot = decide(settings, request.k, ticket, sweep=True)
            if not ahead and at_head is None:
                at_head = time.monotonic()
            if reason is None:
                # Decided again under the lock, without the sweep that may take seconds.
                with exclusive(settings.admit_lock, with_grace(deadline), "admit"):
                    reason, ahead, snapshot = decide(settings, request.k, ticket, sweep=False)
                    if reason is None and snapshot is not None:
                        admitted = dict(
                            record,
                            admitted=round(time.time(), 3),
                            pressure=snapshot.pressure,
                            free=snapshot.free,
                            swapout=None if snapshot.swapout is None else round(snapshot.swapout, 1),
                            running=snapshot.used,
                            head_s=0.0 if at_head is None else round(time.monotonic() - at_head, 3),
                            backfilled=ahead > 0,
                            headroom=snapshot.used + request.k > settings.tokens,
                            ramp=snapshot.used > 0 and settings.ramp > 0 and settings.ramps(snapshot.free),
                        )
                        if held_back:
                            admitted["held_back"] = held_back
                        job = publish_locked(settings.jobs, f"{request.id}.lock", admitted)
                        settings.last_admit.touch()
                        return job, admitted, time.monotonic() - request.started
            progress.reason = held_back = reason if snapshot is None else f"{reason} ({snapshot.memory_text()})"
            if deadline is not None and time.monotonic() >= deadline:
                raise WaitExpired(progress.reason)
            progress.announce()
            if ahead and time.monotonic() >= next_sweep:
                # A head that may not signal a dead gate's survivors (a sandboxed one)
                # would wait on them forever, so every waiter collects them too.
                scan_jobs(settings, sweep=True)
                next_sweep = time.monotonic() + SWEEP_SECONDS
            time.sleep(POLL_SECONDS if not ahead else WAITER_POLL_SECONDS)


def admit_docker(settings: Settings, request: Request, progress: Progress) -> None:
    """Queue for the Docker VM memory the request declares; on return, `request.docker_fd` holds it.

    A reservation counts until its run ends, even once the VM shows the run
    using it: memory counted twice delays a Docker run, while memory counted
    too little starts one that the VM's out-of-memory killer then stops, with
    whatever else runs there.
    """
    record = dict(job_record(request, tokens=False), frees=request.docker_frees or None)
    def reading_limit() -> Limit:
        # The calls of one reading share what the wait has left, but at least DOCKER_MIN_SECONDS.
        if request.deadline is None:
            return lambda: DOCKER_SECONDS
        end = max(request.deadline, time.monotonic() + DOCKER_MIN_SECONDS)
        return lambda: min(DOCKER_SECONDS, end - time.monotonic())

    users, surveyed = "", -math.inf
    # Since when every survey has found the claim more than the VM could offer, and the most it found.
    short_since: Optional[float] = None
    most = 0.0
    with queued(settings, settings.docker, request, record, progress) as ticket:
        wait_for_turn(settings, settings.docker_waiters, ticket, request.deadline, progress)
        while True:
            check_off(settings)
            # A dead gate's survivors may hold a reservation: stop them as the token queue's head would.
            scan_jobs(settings, sweep=True)
            holders = docker_holders(settings, collect=True)
            vm: Dict[str, object] = {}
            survey = time.monotonic() - surveyed >= PROGRESS_SECONDS
            reason = docker_refusal(settings, request, holders, vm, reading_limit(), survey)
            save_vm(settings, vm)
            ceiling = vm.get("ceiling_gib")
            if reason is not None and isinstance(ceiling, float):
                if request.docker_gib <= ceiling:
                    short_since = None
                else:
                    most = ceiling if short_since is None else max(most, ceiling)
                    short_since = time.monotonic() if short_since is None else short_since
                    if time.monotonic() - short_since >= settings.docker_ceiling:
                        raise ConfigError(
                            f"--docker-gib {request.docker_gib:g} exceeds what the Docker VM could offer with "
                            f"every container stopped and every Docker run ended: at most {most:.2f} GiB "
                            f"over {time.monotonic() - short_since:.0f} s"
                        )
            if reason is None:
                admitted = dict(record, admitted=round(time.time(), 3), vm=vm)
                request.docker_fd = publish_locked(settings.docker_jobs, f"{request.id}.lock", admitted)
                request.docker_waited = time.monotonic() - request.started
                return
            if survey:
                # Attempted is enough: a listing that keeps failing must not turn every poll into one.
                surveyed = time.monotonic()
            if "users" in vm:
                users = f"; largest users: {vm['users']}"
            progress.reason = reason + (users if "available_gib" in vm else "")
            if request.deadline is not None and time.monotonic() >= request.deadline:
                raise WaitExpired(progress.reason)
            progress.announce()
            pause = DOCKER_POLL_SECONDS
            if request.deadline is not None:
                pause = min(pause, max(0.0, request.deadline - time.monotonic()))
            # In short naps, so switching the gate off frees this request at once.
            until = time.monotonic() + pause
            while time.monotonic() < until:
                time.sleep(min(WAITER_POLL_SECONDS, max(0.0, until - time.monotonic())))
                check_off(settings)


def release_docker(settings: Settings, request: Request) -> None:
    """Give back the request's Docker VM reservation; one an escaped process still holds stays counted."""
    if request.docker_fd is None:
        return
    os.close(request.docker_fd)
    request.docker_fd = None
    _remove_unlocked(settings.docker_jobs / f"{request.id}.lock")


# ------------------------------------------------------------------ running


def tree_fingerprint(cwd: str, budget: float = 2.0) -> Optional[str]:
    """Identity of the working tree (HEAD, diff, untracked files) without writing to it."""
    deadline = time.monotonic() + budget
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args: str, data: Optional[bytes] = None) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("git", budget)
        return subprocess.run(
            ["git", *args], cwd=cwd, env=env, input=data, capture_output=True, timeout=remaining, check=True
        ).stdout

    try:
        digest = hashlib.sha256(git("rev-parse", "HEAD"))
        digest.update(git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD"))
        untracked = git("ls-files", "-o", "--exclude-standard", "-z")
        digest.update(untracked)
        if untracked:
            paths = untracked.rstrip(b"\0").split(b"\0")
            digest.update(git("hash-object", "--no-filters", "--stdin-paths", data=b"\n".join(paths) + b"\n"))
        return digest.hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
        return None


def cannot_start(program: str, error: OSError) -> int:
    """Report a command that could not be started; its exit code: 126 when not permitted, else 127."""
    say(f"cannot start {program}: {error.strerror or error}")
    return 126 if isinstance(error, PermissionError) else 127


def execute(command: Sequence[str], env: Mapping[str, str]) -> int:
    """Replace the gate with the command; the exit code only when it could not be started."""
    try:
        os.execvpe(command[0], command, env)
    except OSError as error:
        return cannot_start(command[0], error)


def runner_environment(environ: Mapping[str, str], k: int, run_id: str) -> Dict[str, str]:
    """The command's environment: its worker limits and the run tag."""
    env = dict(environ)
    env[CPUS_ENV] = str(k)
    for name in RUNNER_ENV:
        env[name] = str(k)
    env[RUN_ENV] = run_id
    return env


def _tty_delivered_interrupt() -> bool:
    """A terminal Ctrl-C reaches our whole foreground group, the command included."""
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    except OSError:
        return False
    try:
        return os.tcgetpgrp(fd) == os.getpgrp()
    except OSError:
        return False
    finally:
        os.close(fd)


class Stops:
    """INT, TERM and HUP from admission until the run's released event is logged.

    While the command runs, they reach it (a terminal's Ctrl-C has reached it
    already). One that comes before it starts means it never does; one that
    comes while it starts reaches it once it has. After it exits they change
    nothing, so the gate still sweeps, releases and logs the run.
    One the caller ignores stays ignored, for the command too.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.child: Optional["subprocess.Popen[bytes]"] = None
        #: Set once the command has exited and been reaped.
        self.exited = threading.Event()
        #: Monotonic time of the first stop request before the command exited.
        self.first: Optional[float] = None
        self._early: List[int] = []
        self._previous: Dict[int, object] = {}

    def __enter__(self) -> "Stops":
        # One the caller ignores, as under nohup, stays ignored for the gate and the command.
        self._previous = {
            signum: signal.signal(signum, self._receive)
            for signum in self.SIGNALS
            if signal.getsignal(signum) is not signal.SIG_IGN
        }
        return self

    def __exit__(self, *_exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]

    def _receive(self, signum: int, _frame: object) -> None:
        if self.exited.is_set():
            return
        if self.child is None:
            self._early.append(signum)
        elif not (signum == signal.SIGINT and _tty_delivered_interrupt()):
            self.send(signum)
        if self.first is None:
            self.first = time.monotonic()

    def before_start(self) -> Optional[int]:
        """The first stop request so far if the command has not started, else None."""
        return self._early[0] if self.child is None and self._early else None

    def started(self, child: "subprocess.Popen[bytes]") -> None:
        self.child = child
        for signum in self._early:
            self.send(signum)

    def send(self, signum: int) -> None:
        if self.child is not None and not self.exited.is_set():
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.child.pid, signum)


def run_admitted(
    settings: Settings, request: Request, job_fd: int, record: Dict[str, object], waited: float
) -> int:
    with Stops() as stops:
        return _run_admitted(settings, request, job_fd, record, waited, stops)


def _run_admitted(
    settings: Settings, request: Request, job_fd: int, record: Dict[str, object], waited: float, stops: Stops
) -> int:
    run_id = request.id
    log_event(settings, dict(record, event="admitted", waited_s=round(waited, 3)))
    fingerprint: Dict[str, Optional[str]] = {}
    fingerprinter = threading.Thread(
        target=lambda: fingerprint.update(tree=tree_fingerprint(os.getcwd())), daemon=True
    )
    fingerprinter.start()
    env = runner_environment(settings.environ, request.k, run_id)
    began = time.monotonic()
    host_before = host_cpu_ticks()
    stopped = stops.before_start()
    if stopped is not None:
        # Stopped before it started: nothing runs, as when stopped while queued.
        _release(settings, run_id, job_fd)
        release_docker(settings, request)
        log_event(settings, {
            "event": "released", "id": run_id, "label": request.label, "cwd": record.get("cwd"),
            "k": request.k, "asked": request.asked, "docker_gib": request.docker_gib,
            "waited_s": round(waited, 3), "docker_waited_s": round(request.docker_waited, 3),
            "held_s": round(time.monotonic() - began, 3), "exit": 128 + stopped, "signal": None,
            "timed_out": False, "cancelled": True, "cpu_s": None, "cpus": None, "swept": 0, "survivors": 0,
            "left_held": False,
        })
        return 128 + stopped
    try:
        # The command inherits every reservation, so they outlive a killed gate.
        held_fds = (job_fd,) + (() if request.docker_fd is None else (request.docker_fd,))
        child = subprocess.Popen(request.argv, env=env, pass_fds=held_fds)
        yield_to_the_owner(child.pid)
    except OSError as error:
        code = cannot_start(request.argv[0], error)
        _release(settings, run_id, job_fd)
        release_docker(settings, request)
        return code
    stops.started(child)

    used, watched = supervise(settings, request, child, record, began, stops)
    held = time.monotonic() - began
    busy = host_busy(host_before, host_cpu_ticks())
    swept, survivors = sweep_run(run_id)
    # Survivors keep the run's tokens: its record stays for the queue head to collect.
    left_held = _release(settings, run_id, job_fd, keep=survivors != {})
    release_docker(settings, request)
    fingerprinter.join(0.1)
    code = child.returncode
    exit_code = TIMEOUT_EXIT if watched["timed_out"] else (128 - code if code < 0 else code)
    log_event(
        settings,
        {
            "event": "released",
            "id": run_id,
            "label": request.label,
            "cwd": record.get("cwd"),
            "k": request.k,
            "asked": request.asked,
            "docker_gib": request.docker_gib,
            "waited_s": round(waited, 3),
            "docker_waited_s": round(request.docker_waited, 3),
            "held_s": round(held, 3),
            "exit": exit_code,
            "signal": -code if code < 0 else None,
            "timed_out": watched["timed_out"],
            "cancelled": watched["cancelled"],
            "pressure": record.get("pressure"),
            "free": record.get("free"),
            "running": record.get("running"),
            "min_free": watched["min_free"],
            "max_pressure": watched["max_pressure"],
            "max_swapout": watched["max_swapout"],
            "children_maxrss_mib": round(
                used.get("maxrss", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2**20, 1
            ),
            "cpu_s": round(used["cpu_s"], 2) if "cpu_s" in used else None,
            "cpus": round(used["cpu_s"] / held, 2) if "cpu_s" in used and held > 0 else None,
            "host_busy": None if busy is None else round(busy, 3),
            # What everything else on the host used meanwhile: agents, servers, the Docker VM, other runs.
            "others_cpus": None if busy is None or "cpu_s" not in used or held <= 0
            else round(max(0.0, busy * (os.cpu_count() or 1) - used["cpu_s"] / held), 2),
            "swept": swept,
            "survivors": None if survivors is None else len(survivors),
            "left_held": left_held,
            "tree": fingerprint.get("tree"),
        },
    )
    return exit_code


def supervise(settings: Settings, request: Request, child: "subprocess.Popen[bytes]", record: Mapping[str, object],
              began: float, stops: Stops) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Wait for the command to exit while the gate guards it.

    A command still running CANCEL_SECONDS after the first stop request (see
    `Stops`) gets TERM, then KILL after GRACE_SECONDS. The --timeout fuse
    sends TERM, then KILL after GRACE_SECONDS. Memory is sampled every
    SAMPLE_SECONDS. Returns what its tree used (see `reap`) and the released
    event's timed_out, cancelled, min_free and max_pressure.
    """
    used: Dict[str, float] = {}

    def waiter() -> None:
        try:
            used.update(reap(child))
        finally:
            stops.exited.set()

    threading.Thread(target=waiter, daemon=True).start()
    escalate: Dict[str, float] = {}
    cancel_scheduled = False
    timed_out = False
    min_free, max_pressure, max_swapout = record.get("free"), record.get("pressure"), record.get("swapout")
    next_sample = began + SAMPLE_SECONDS
    while not stops.exited.wait(0.5):
        now = time.monotonic()
        if stops.first is not None and not cancel_scheduled:
            cancel_scheduled = True
            escalate["term"] = stops.first + CANCEL_SECONDS
        if now >= next_sample:
            next_sample = now + SAMPLE_SECONDS
            pressure, free, swapout = memory_signal(settings.environ)
            if free is not None and (min_free is None or free < number(min_free)):
                min_free = free
            if pressure is not None and (max_pressure is None or pressure > number(max_pressure)):
                max_pressure = pressure
            if swapout is not None and (max_swapout is None or swapout > number(max_swapout)):
                max_swapout = round(swapout, 1)
        if request.timeout and not timed_out and now - began >= request.timeout:
            timed_out = True
            say(f"{request.label} ran past its {request.timeout:g}s fuse; stopping it")
            stops.send(signal.SIGTERM)
            escalate["kill"] = now + GRACE_SECONDS
        if "term" in escalate and now >= escalate["term"]:
            del escalate["term"]
            stops.send(signal.SIGTERM)
            escalate["kill"] = now + GRACE_SECONDS
        if "kill" in escalate and now >= escalate["kill"]:
            del escalate["kill"]
            stops.send(signal.SIGKILL)
    return used, {"timed_out": timed_out, "cancelled": stops.first is not None, "min_free": min_free,
                  "max_pressure": max_pressure, "max_swapout": max_swapout}


def reap(child: "subprocess.Popen[bytes]") -> Dict[str, float]:
    """Wait for the command; return the CPU seconds and peak RSS its tree used.

    wait4 reports the command and every descendant reaped beneath it, and none of
    the gate's own probes (ps, lsof, git). A process that detached and was never
    reaped there is missing, so the figure is a floor for such runs.
    """
    try:
        _pid, wait_status, rusage = os.wait4(child.pid, 0)
    except OSError:  # Popen reaps it and reports its own exit; there is no usage
        child.wait()
        return {}
    child.returncode = os.waitstatus_to_exitcode(wait_status)
    return {"cpu_s": rusage.ru_utime + rusage.ru_stime, "maxrss": float(rusage.ru_maxrss)}


def _release(settings: Settings, run_id: str, job_fd: int, keep: bool = False) -> bool:
    """Drop the job lock and its file; True when the run's record stays counted.

    It stays when `keep` is set or an escaped process still holds the job
    lock. The queue head stops whatever remains once this gate is gone.
    """
    os.close(job_fd)
    if keep:
        return True
    path = settings.jobs / f"{run_id}.lock"
    for _ in range(20):
        if _remove_unlocked(path):
            return False
        time.sleep(0.1)
    return True


# ------------------------------------------------------------------ commands


def capped_claim(settings: Settings, asked: int) -> int:
    """The tokens a run holds for `asked`: no more than the per-run cap (AGENT_GATE_MAX_CLAIM)."""
    return min(asked, settings.max_claim)


def run(argv: Sequence[str], environ: Optional[Mapping[str, str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-gate", description="Run a heavy command when the machine has room.")
    parser.add_argument("-c", "--cpus", default=None,
                        help=f"tokens to hold (default {DEFAULT_CPUS}; at most {MAX_CLAIM_ENV}, half the tokens)")
    parser.add_argument("-l", "--label", default="")
    parser.add_argument("--docker-gib", default="0", metavar="G",
                        help="memory the command's containers need in the Docker VM, in GiB")
    parser.add_argument("--docker-frees", default="", metavar="PROJECT",
                        help="a compose project the command stops first; what it uses counts as available")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--wait", default=None, help="give up after SECS in the queue (exit 75)")
    group.add_argument("--no-wait", action="store_true", help="exit 75 unless admitted at once")
    parser.add_argument("--timeout", default=str(DEFAULT_TIMEOUT), help="stop the run after SECS (0: never)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    try:
        settings = load_settings(environ)
        environ = settings.environ
        asked = DEFAULT_CPUS if args.cpus is None else math.ceil(finite(args.cpus, "-c"))
        if asked < 1:
            raise ConfigError(f"-c {args.cpus} is below 1 token")
        k = capped_claim(settings, asked)
        docker_gib = finite(args.docker_gib, "--docker-gib")
        if args.docker_frees and not docker_gib:
            raise ConfigError("--docker-frees needs --docker-gib")
        wait = 0.0 if args.no_wait else None if args.wait is None else finite(args.wait, "--wait")
        timeout = finite(args.timeout, "--timeout")
        if settings.off.exists():
            return execute(command, environ)
        settings.ensure()
        label = args.label or " ".join(Path(command[0]).name.split() + command[1:3])
        parent = enclosing_run(settings, environ)
    except (ConfigError, OSError) as error:
        return unusable(error)

    if parent is None and k < asked and args.cpus is not None:
        say(f"{label} asks for {asked} tokens and runs with {k}, the most one run holds on this machine")
    if parent is not None:
        # Nested: the parent already holds the budget; waiting here could deadlock.
        budget = min(k, int(number(parent.get("k"), k)))
        reserved = number(parent.get("docker_gib"))
        if docker_gib > reserved:
            say(f"{label} needs {docker_gib:g} GiB in the Docker VM inside a run that reserved {reserved:g}; "
                "it starts at once, so the outer run should declare it")
        log_event(settings, {"event": "nested", "parent": parent.get("id"), "label": label, "k": budget,
                             "asked": asked, "docker_gib": docker_gib, "cwd": os.getcwd()})
        return execute(command, runner_environment(environ, budget, str(parent["id"])))

    request = Request(k, label, command, wait, timeout, asked, docker_gib, args.docker_frees)

    def not_admitted(reason: str, code: int) -> int:
        log_event(settings, {"event": "not-admitted", "id": request.id, "label": label, "reason": reason,
                             "docker_gib": docker_gib})
        return code

    progresses: List[Progress] = []
    try:
        # Docker first: a run waiting for the VM holds no tokens, so it delays no other run.
        if docker_gib:
            progresses.append(Progress(settings, label, k, docker_gib))
            admit_docker(settings, request, progresses[-1])
        progresses.append(Progress(settings, label, k))
        job_fd, record, waited = admit(settings, request, progresses[-1])
    except SwitchedOff:
        release_docker(settings, request)
        say(f"admission control is off: starting {label} after {time.monotonic() - request.started:.0f}s")
        return execute(command, environ)
    except WaitExpired as error:
        release_docker(settings, request)
        say(f"not admitted: {error}; nothing ran")
        return not_admitted(str(error), NOT_ADMITTED_EXIT)
    except KeyboardInterrupt:
        release_docker(settings, request)
        return not_admitted("interrupted", 130)
    except (ConfigError, OSError) as error:
        release_docker(settings, request)
        return not_admitted(f"unusable: {error}", unusable(error))
    if any(progress.announced for progress in progresses):
        say(f"admitted {label} after {waited:.0f}s")
    return run_admitted(settings, request, job_fd, record, waited)


def switch(on: bool, environ: Optional[Mapping[str, str]] = None) -> int:
    """`agent-gate on|off`: resume or stop admission control for every gate on the host."""
    try:
        settings = load_settings(environ)
        settings.ensure()
        if on:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(settings.off)
        else:
            settings.off.touch()
    except (ConfigError, OSError) as error:
        return unusable(error)
    say("admission control is on" if on else "admission control is off: every run starts at once")
    return 0


def inside(argv: Sequence[str] = (), environ: Optional[Mapping[str, str]] = None) -> int:
    """Exit 0 inside a live run whose budget covers `-c` or while the gate is off, 1 outside, 78 when it does not cover it.

    `-c` is judged after the cap a claim gets (AGENT_GATE_MAX_CLAIM): a caller that asked for more than the
    cap was admitted with the cap, so asking `inside` for the same amount must find its own run covering it.
    """
    parser = argparse.ArgumentParser(prog="agent-gate inside")
    parser.add_argument("-c", "--cpus", default=None,
                        help=f"tokens the caller needs from the enclosing run (at most {MAX_CLAIM_ENV}, as for a claim)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(environ)
        environ = settings.environ
        need = 0 if args.cpus is None else capped_claim(settings, math.ceil(finite(args.cpus, "-c")))
        if settings.off.exists():
            return 0
        parent = enclosing_run(settings, environ)
        if parent is not None and need > number(parent.get("k")):
            raise ConfigError(f"-c {args.cpus} (as {need} tokens) exceeds the enclosing run's {parent.get('k')} tokens")
    except (ConfigError, OSError) as error:
        return unusable(error)
    return 0 if parent is not None else 1


# ------------------------------------------------------------------ reports


def status(argv: Sequence[str], environ: Optional[Mapping[str, str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-gate status")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(environ)
        # A missing directory reads as empty; one that cannot be read is unusable.
        snapshot = Snapshot(settings, sweep=False)
        waiters = live_waiters(settings, collect=False)
        docker_reserved = docker_holders(settings)
        docker_waiting = live_waiters(settings, collect=False, directory=settings.docker_waiters)
        off = settings.off.exists()
    except (ConfigError, OSError) as error:
        return unusable(error)
    try:
        vm: Optional[Dict[str, object]] = json.loads(settings.docker_vm.read_text())
    except (OSError, ValueError):
        vm = None
    now = time.time()
    # Requests that have waited past the backfill bound: the tokens they claim go to no later
    # request, which passes them only as a 1-token run into the headroom while runs outside
    # the headroom hold all T tokens.
    aged = [waiter for waiter in waiters if is_aged(waiter, settings, now)]
    reserved = sum(number(waiter.get("k")) for waiter in aged)
    if args.json:
        print(json.dumps({
            "docker": {"vm": vm, "reserved": docker_reserved, "waiting": docker_waiting},
            "tokens": settings.tokens,
            "headroom": settings.headroom,
            "used": snapshot.used,
            "outside_headroom": snapshot.outside_headroom,
            "headroom_used": snapshot.in_headroom,
            "reserved": reserved,
            "backfill_s": settings.backfill,
            "pressure": snapshot.pressure,
            "free": snapshot.free,
            "swapout": snapshot.swapout,
            "min_free": settings.min_free,
            "running": snapshot.jobs,
            "waiting": waiters,
            "off": off,
        }, indent=2, sort_keys=True))
        return 0
    if off:
        print("admission control is OFF (`agent-gate on` resumes it): every run starts at once")
    headroom = ""
    if settings.headroom:
        headroom = f" (+{settings.headroom} for 1-token runs; runs admitted into it hold {snapshot.in_headroom:g})"
    print(
        f"{snapshot.used:g} of {settings.tokens} tokens in use{headroom}; {snapshot.memory_text()} "
        f"(new runs need {settings.min_free:g}% free while others run)"
    )
    if aged:
        print(f"{reserved:g} tokens reserved for {len(aged)} request(s) queued over {settings.backfill:g}s; "
              f"only a 1-token run into the headroom passes them, while runs outside it hold all "
              f"{settings.tokens} tokens ({snapshot.outside_headroom:g} now)")
    for job in snapshot.jobs:
        age = now - number(job.get("admitted"), now)
        note = "" if job["gate_alive"] else "  [gate gone: orphaned]" if job["held"] else "  [gate gone: leftovers]"
        print(f"  running k={job.get('k')} {age:6.0f}s  {job.get('label')}  {job.get('cwd')}{note}")
    for position, waiter in enumerate(waiters, 1):
        age = now - number(waiter.get("queued"), now)
        print(f"  queued #{position} k={waiter.get('k')} {age:6.0f}s  {waiter.get('label')}  {waiter.get('cwd')}")
    recent = vm is not None and now - number(vm.get("at")) < VM_READING_SECONDS
    if recent or docker_reserved or docker_waiting:
        vm_reserved = reserved_gib(docker_reserved)
        if vm is None:
            reading = "not read yet"
        elif "available_gib" in vm:
            reading = f"{vm['available_gib']:g} of {vm['total_gib']:g} GiB available"
        else:
            reading = f"unreadable ({vm.get('error')})"
        age = f" {now - number(vm.get('at'), now):.0f}s ago" if vm else ""
        print(f"Docker VM: {reading}{age}; {vm_reserved:g} GiB reserved")
        for holder in docker_reserved:
            print(f"  reserved {number(holder.get('docker_gib')):g} GiB  {holder.get('label')}  {holder.get('cwd')}")
        for position, waiter in enumerate(docker_waiting, 1):
            age_s = now - number(waiter.get("queued"), now)
            print(f"  queued #{position} {number(waiter.get('docker_gib')):g} GiB {age_s:6.0f}s  "
                  f"{waiter.get('label')}  {waiter.get('cwd')}")
    return 0


def command_kind(label: str) -> str:
    """A label with its path operands folded, so runs of one command group together."""
    words: List[str] = []
    for word in label.split():
        folded = "…" if "/" in word or re.search(r"\.(py|[cm]?[jt]sx?|json|spec|test)$", word) else word
        if not (folded == "…" and words and words[-1] == "…"):
            words.append(folded)
    return " ".join(words)


def percentile(values: Sequence[float], share: float, weights: Optional[Sequence[float]] = None) -> float:
    """The value below which `share` of the weight lies; each value weighs 1 without weights."""
    pairs = sorted(zip(values, weights if weights is not None else [1.0] * len(values)))
    total = sum(weight for _value, weight in pairs)
    reached = 0.0
    for value, weight in pairs:
        reached += weight
        if reached >= share * total:
            return value
    return pairs[-1][0]


def others_text(cpus: object) -> str:
    return "-" if cpus is None else f"{number(cpus):.1f}"


def measured_runs(settings: Settings, since: float, match: str) -> Dict[Tuple[str, int], List[Dict[str, object]]]:
    """Released runs that recorded their CPUs since `since`, labelled with `match`, by kind of command and claim.

    A log not written yet holds none; OSError when the state directory cannot be read.
    """
    kinds: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    for path in (settings.events.with_name(settings.events.name + ".1"), settings.events):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            label = str(event.get("label") or "")
            if (
                event.get("event") != "released"
                or event.get("cpus") is None
                or number(event.get("ts"), 0) < since
                or match not in label
            ):
                continue
            kinds.setdefault((command_kind(label), int(number(event.get("k"), 0))), []).append(event)
    return kinds


def usage_row(kind: str, k: int, events: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """One kind of command's claim `k` against the CPUs its runs used, weighed by held time."""
    cpus = [number(event["cpus"], 0) for event in events]
    held = [number(event.get("held_s"), 0) for event in events]
    rss = [number(event.get("children_maxrss_mib"), 0) for event in events]
    weights = [max(seconds, 0.001) for seconds in held]
    p90 = percentile(cpus, 0.9, weights)
    in_docker = any(number(event.get("docker_gib"), 0) > 0 for event in events)
    others = [(number(event["others_cpus"], 0), weight) for event, weight in zip(events, weights)
              if event.get("others_cpus") is not None]
    return {
        "kind": kind,
        "k": k,
        "runs": len(events),
        "cpus_p50": round(percentile(cpus, 0.5, weights), 2),
        "cpus_p90": round(p90, 2),
        "held_s_p50": round(percentile(held, 0.5), 1),
        "token_s": round(k * sum(held), 1),
        "maxrss_mib_p90": round(percentile(rss, 0.9), 1),
        "docker": in_docker,
        "fits": None if in_docker else max(1, math.floor(p90 + 0.5)),
        "others_cpus_p50": round(percentile([value for value, _ in others], 0.5,
                                            [weight for _, weight in others]), 2) if others else None,
    }


def usage(argv: Sequence[str], environ: Optional[Mapping[str, str]] = None) -> int:
    """`agent-gate usage`: compare each command kind's token claims with measured CPUs."""
    parser = argparse.ArgumentParser(prog="agent-gate usage")
    parser.add_argument("--hours", default="72", help="the window, in hours back from now")
    parser.add_argument("--match", default="", help="only labels containing this text")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(environ)
        hours = finite(args.hours, "--hours")
        kinds = measured_runs(settings, time.time() - hours * 3600, args.match)
    except (ConfigError, OSError) as error:
        return unusable(error)
    rows = [usage_row(kind, k, events) for (kind, k), events in kinds.items()]
    rows.sort(key=lambda row: -number(row["token_s"], 0))
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("no measured runs in the window (runs record CPUs from this version on)")
        return 0
    print(f"{'runs':>4} {'k':>2} {'fits':>4} {'cpus p50':>8} {'p90':>5} {'held p50':>8} {'rss p90':>8} {'others':>6}  command")
    for row in rows:
        fits = row["fits"]
        mark = "  " if fits is None or fits == row["k"] else " <" if fits < row["k"] else " >"
        print(
            f"{row['runs']:>4} {row['k']:>2} {'vm' if fits is None else fits:>4}{mark}{row['cpus_p50']:>6.2f} {row['cpus_p90']:>5.2f} "
            f"{row['held_s_p50']:>7.1f}s {row['maxrss_mib_p90']:>6.0f}Mi {others_text(row['others_cpus_p50']):>6}  {row['kind']}"
        )
    print(
        "fits: the time-weighted p90 CPUs to the nearest token; < claims more than it uses, > less. "
        "Detached processes are not counted; vm: Docker containers' CPUs are not either. "
        "others: the CPUs the rest of the host used meanwhile (p50), agents, servers and the Docker VM included; "
        "when they and the run's CPUs fill the host, the run got less than it would use. "
        "An average also hides the bursts a parallel phase needs its claim for."
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["status"]:
        return status(argv[1:])
    if argv[:1] == ["inside"]:
        return inside(argv[1:])
    if argv[:1] == ["usage"]:
        return usage(argv[1:])
    if argv in (["on"], ["off"]):
        return switch(argv == ["on"])
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
