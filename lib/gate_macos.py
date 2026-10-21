"""macOS process inspection and machine signals used by agent-gate."""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

RUN_ENV = "AGENT_GATE_RUN"
# Tests can supply a file with pressure, free-memory share and swap-out rate.
MEMORY_FILE_ENV = "AGENT_GATE_MEMORY_FILE"
# The swap-out rate uses readings taken over this window.
SWAP_MIN_SECONDS = 1.0
SWAP_WINDOW_SECONDS = 10.0
# A gated command yields CPU time to interactive applications.
RUN_NICENESS = 10
GRACE_SECONDS = 5.0
# Process start times have one-second resolution.
START_TOLERANCE_SECONDS = 1
# Give a process in exec time to make its arguments readable.
EXEC_SECONDS = 1.0
# By absolute path: callers' PATH often leaves out /usr/sbin.
LSOF = "/usr/sbin/lsof"

_LIBC = ctypes.CDLL(None, use_errno=True)


def sysctl_int(name: str, kind: "type" = ctypes.c_int) -> Optional[int]:
    """One integer sysctl, read as the ctypes type `kind` (c_int, or c_int64 for sizes); None when unreadable."""
    value = kind(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    result = _LIBC.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0))
    return value.value if result == 0 else None


class _BsdInfo(ctypes.Structure):
    """struct proc_bsdinfo from <sys/proc_info.h>."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


# Process data comes from libproc and sysctl directly. /bin/ps is setuid, and
# the Codex sandbox refuses to execute setuid binaries.
_PROC_PIDTBSDINFO = 3
_SZOMB = 5
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_proc_pidinfo = _LIBC.proc_pidinfo
_proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
_proc_pidinfo.restype = ctypes.c_int
_LIBC.sysctl.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p, ctypes.c_size_t,
]
_proc_listallpids = _LIBC.proc_listallpids
_proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
_proc_listallpids.restype = ctypes.c_int


def process_info(pid: object) -> Optional[Tuple[int, int]]:
    """(parent pid, start second) of a live process of this user; None when gone or a zombie.

    Other users' processes are unreadable and also yield None; every process the
    gate tracks belongs to this user.
    """
    try:
        pid = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    info = _BsdInfo()
    if _proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return None
    return None if info.pbi_status == _SZOMB else (int(info.pbi_ppid), int(info.pbi_start_tvsec))


def all_pids() -> Optional[List[int]]:
    count = _proc_listallpids(None, 0)
    if count <= 0:
        return None
    buffer = (ctypes.c_int * (count + 256))()
    count = _proc_listallpids(buffer, ctypes.sizeof(buffer))
    return list(buffer[:count]) if count > 0 else None


def _procargs(pid: int, buffer: "ctypes.Array[ctypes.c_char]") -> Optional[Tuple[List[str], List[str]]]:
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(len(buffer))
    if _LIBC.sysctl(mib, 3, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0 or size.value < 4:
        return None
    data = buffer.raw[: size.value]
    argc = int.from_bytes(data[:4], sys.byteorder)
    # argc, the executable path, NUL padding, then argv and environment strings.
    start = data.find(b"\0", 4)
    if start < 0:
        return None
    while start < len(data) and data[start] == 0:
        start += 1
    strings = [item.decode(errors="replace") for item in data[start:].split(b"\0")]
    environment = []
    for item in strings[argc:]:
        if not item:
            break
        environment.append(item)
    return strings[:argc], environment


def _args_buffer() -> "ctypes.Array[ctypes.c_char]":
    return ctypes.create_string_buffer(sysctl_int("kern.argmax") or 2**20)


def own_start() -> int:
    info = process_info(os.getpid())
    return info[1] if info else 0


def same_process(pid: object, start: object) -> bool:
    info = process_info(pid)
    if info is None:
        return False
    try:
        return abs(info[1] - float(start)) <= START_TOLERANCE_SECONDS
    except (TypeError, ValueError):
        return False


def ancestors() -> Dict[int, int]:
    """Every ancestor of this process: pid -> start second."""
    chain: Dict[int, int] = {}
    current = os.getppid()
    while current > 1 and current not in chain and len(chain) < 128:
        info = process_info(current)
        if info is None:
            break
        chain[current] = info[1]
        current = info[0]
    return chain


#: A process identity: pid and start second, so a recycled pid is never mistaken for it.
Processes = Dict[int, int]


def identify(pids: Iterable[int]) -> Processes:
    found: Processes = {}
    for pid in pids:
        info = process_info(pid)
        if info is not None and pid > 1 and pid != os.getpid():
            found[pid] = info[1]
    return found


def tagged_pids(run_id: str) -> Optional[Processes]:
    """Processes whose environment carries this run's tag; None when the table is unreadable.

    Platform binaries (sh, sleep) show no environment, so they are invisible
    here; runners, browsers and servers are not. A process in the middle of
    exec has no readable arguments for a moment, and a launcher that execs the
    real program (framework Python's does) gets there after it started, so
    such a process is read again until its exec is done.
    """
    pids = all_pids()
    if pids is None:
        return None
    needle = f"{RUN_ENV}={run_id}"
    buffer = _args_buffer()
    found: Processes = {}
    pending = [pid for pid in pids if pid > 1 and pid != os.getpid()]
    # Counted from the end of the first pass, which a loaded host may make long.
    deadline: Optional[float] = None
    while True:
        in_exec = []
        for pid in pending:
            # The identity is read on both sides of the environment, so a pid
            # recycled in between is never taken for the tagged process.
            before = process_info(pid)
            if before is None:
                continue
            args = _procargs(pid, buffer)
            if args is None:
                in_exec.append(pid)
                continue
            after = process_info(pid) if needle in args[1] else None
            if after is not None and after[1] == before[1]:
                found[pid] = before[1]
        if deadline is None:
            deadline = time.monotonic() + EXEC_SECONDS
        if not in_exec or time.monotonic() >= deadline:
            return found
        pending = in_exec
        time.sleep(0.01)


def lock_holders(path: Path, since: float) -> Optional[Processes]:
    """Processes that hold `path` open for writing and started after `since`; None when lsof cannot say.

    Only the owning gate opens a job file read-write, and the run's processes
    inherit it; other gates probe it read-only. So a reader, or anything older
    than the run, cannot be one of them. Only a process seen in two listings
    apart counts, so a descriptor in passing is not mistaken for a holder.
    """
    seen: List[set] = []
    for attempt in range(2):
        if attempt:
            time.sleep(0.3)
        try:
            out = subprocess.run(
                [LSOF, "-F", "pa", "--", str(path)], capture_output=True, text=True, timeout=30, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        writers, pid = set(), 0
        for line in out.splitlines():
            if line[:1] == "p" and line[1:].isdigit():
                pid = int(line[1:])
            elif line[:1] == "a" and ("u" in line or "w" in line):
                writers.add(pid)
        # Identified right after each listing: a pid recycled after the first
        # listing shows another start time after the second one.
        seen.append(set(identify(writers).items()))
    return {pid: start for pid, start in seen[0] & seen[1] if start >= since - 1}


def stop_processes(targets: Processes, still_owned: Callable[[], Optional[Processes]]) -> int:
    """TERM, wait GRACE_SECONDS, then KILL whatever `still_owned` still reports.

    Every signal goes to a pid only while it still names the same process.
    Returns how many processes took the TERM. A sandbox may refuse to signal
    processes outside it; those are left to a gate outside the sandbox.
    """
    termed: Processes = {}
    for pid, start in sorted(targets.items()):
        if pid > 1 and pid != os.getpid() and identify([pid]).get(pid) == start:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)
                termed[pid] = start
    deadline = time.monotonic() + GRACE_SECONDS
    alive = dict(termed)
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        current = identify(alive)
        alive = {pid: start for pid, start in alive.items() if current.get(pid) == start}
    if alive:
        owned = still_owned() or {}
        for pid, start in alive.items():
            if owned.get(pid) == start and identify([pid]).get(pid) == start:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)
    return len(termed)


def sweep_run(run_id: str) -> Tuple[int, Optional[Processes]]:
    """Stop processes left behind by a run: (how many took a TERM, what survived).

    A leftover may start a replacement while it stops, so the sweep repeats
    while new tagged processes appear. The survivors are None when the process
    table is unreadable.
    """
    stopped = 0
    for _ in range(3):
        pids = tagged_pids(run_id)
        if not pids:
            return stopped, pids
        stopped += stop_processes(pids, lambda: tagged_pids(run_id))
    return stopped, tagged_pids(run_id)


class _VmStatistics64(ctypes.Structure):
    """struct vm_statistics64 from <mach/vm_statistics.h>."""

    _fields_ = (
        [(name, ctypes.c_uint32) for name in ("free_count", "active_count", "inactive_count", "wire_count")]
        + [(name, ctypes.c_uint64) for name in (
            "zero_fill_count", "reactivations", "pageins", "pageouts", "faults", "cow_faults", "lookups", "hits",
            "purges",
        )]
        + [(name, ctypes.c_uint32) for name in ("purgeable_count", "speculative_count")]
        + [(name, ctypes.c_uint64) for name in ("decompressions", "compressions", "swapins", "swapouts")]
        + [(name, ctypes.c_uint32) for name in (
            "compressor_page_count", "throttled_count", "external_page_count", "internal_page_count",
        )]
        + [("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]
    )


class _HostBasicInfo(ctypes.Structure):
    """struct host_basic_info from <mach/host_info.h>."""

    _fields_ = [(name, ctypes.c_int) for name in ("max_cpus", "avail_cpus")] + [
        ("memory_size", ctypes.c_uint32)
    ] + [(name, ctypes.c_int) for name in (
        "cpu_type", "cpu_subtype", "cpu_threadtype", "physical_cpu", "physical_cpu_max", "logical_cpu",
        "logical_cpu_max",
    )] + [("max_mem", ctypes.c_uint64)]


_HOST_BASIC_INFO = 1
_HOST_VM_INFO64 = 4
_HOST_CPU_LOAD_INFO = 3
_host_statistics64 = _LIBC.host_statistics64
_LIBC.mach_host_self.restype = ctypes.c_uint32
_host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_LIBC.host_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_LIBC.host_statistics.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
_host_port = _LIBC.mach_host_self()


def _memory_pages() -> Optional[int]:
    """Pages of usable memory, the denominator of the free share.

    The Codex sandbox refuses hw.memsize_usable; host_info's physical size is
    slightly larger, so the share computed from it never overstates free memory.
    """
    page = sysctl_int("hw.pagesize", ctypes.c_int64)
    usable = sysctl_int("hw.memsize_usable", ctypes.c_int64)
    if usable is None:
        info = _HostBasicInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
        if _LIBC.host_info(_host_port, _HOST_BASIC_INFO, ctypes.byref(info), ctypes.byref(count)) == 0:
            usable = info.max_mem or None
    return usable // page if usable and page else None


_MEMORY_PAGES = _memory_pages()
_PAGE_BYTES = sysctl_int("hw.pagesize", ctypes.c_int64) or 16384


def free_share(stats: "_VmStatistics64", total_pages: Optional[int]) -> Optional[int]:
    """Free, file-backed and purgeable pages over usable memory: what the kernel can hand out
    without compressing or swapping app memory.

    kern.memorystatus_level also counts active and inactive app memory, which the
    compressor can squeeze, so it read 30-44% while the host swapped out 270 MB/s.
    host_statistics64 folds speculative pages, which are file-backed, into
    free_count, so they are taken out of it to count once.
    """
    if not total_pages:
        return None
    free = stats.free_count - stats.speculative_count + stats.external_page_count + stats.purgeable_count
    return min(100, free * 100 // total_pages)


def _readings(samples: Path) -> List[Tuple[float, int]]:
    try:
        value = json.loads(samples.read_text())
        return [(float(at), int(swapped)) for at, swapped in value]
    except (OSError, ValueError, TypeError):
        return []


def swapout_rate(swapped_out: int, samples: Path) -> Optional[float]:
    """MiB/s swapped out over about the last SWAP_WINDOW_SECONDS; None until readings SWAP_MIN_SECONDS apart exist.

    Every gate on the host adds its readings to `samples`, so one that has just started
    sees the rate the others measured; a waiter polls and a running gate samples, so
    while runs run the readings stay fresh.
    """
    now = time.time()
    # A reading from the future was taken before a clock was set back.
    readings = [reading for reading in _readings(samples) if reading[0] <= now]
    newest = readings[-1][0] if readings else -math.inf
    readings.append((now, swapped_out))
    while len(readings) > 2 and now - readings[1][0] >= SWAP_WINDOW_SECONDS:
        del readings[0]
    if now - readings[0][0] > 2 * SWAP_WINDOW_SECONDS:
        readings = readings[-1:]  # an average over a quiet stretch says nothing about now
    if now - newest >= SWAP_MIN_SECONDS / 2:
        staging = samples.with_name(f".{samples.name}.{os.getpid()}.tmp")
        with contextlib.suppress(OSError):
            staging.write_text(json.dumps(readings))
            os.replace(staging, samples)
    then, before = readings[0]
    return None if now - then < SWAP_MIN_SECONDS else max(0, swapped_out - before) / (now - then) / 2**20


def memory_signal(environ: Mapping[str, str],
                  samples: Path) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """(pressure level, free percent, swap-out MiB/s); each is None when unreadable here or not yet measured.

    `samples` keeps the host's recent swap-out readings (see `swapout_rate`).

    The pressure level is unreadable inside the Codex sandbox.
    """
    override = environ.get(MEMORY_FILE_ENV)
    if override:
        try:
            words = Path(override).read_text().split()[:3]
            pressure, free = (None if word == "-" else int(word) for word in words[:2])
            return pressure, free, float(words[2]) if len(words) > 2 else 0.0
        except (OSError, ValueError):
            return None, None, None
    stats = _VmStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    if _host_statistics64(_host_port, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)) != 0:
        return sysctl_int("kern.memorystatus_vm_pressure_level"), None, None
    return (sysctl_int("kern.memorystatus_vm_pressure_level"), free_share(stats, _MEMORY_PAGES),
            swapout_rate(stats.swapouts * _PAGE_BYTES, samples))


#: (busy counters, idle counters, the modulus they wrap at)
Ticks = Tuple[Tuple[int, ...], Tuple[int, ...], int]


def host_cpu_ticks() -> Optional[Ticks]:
    """The host's CPU tick counters since boot, over all CPUs; None where unreadable.

    It only describes a run, so no failure may escape it.
    """
    try:
        load = (ctypes.c_uint32 * 4)()  # user, system, idle, nice
        count = ctypes.c_uint32(4)
        if _LIBC.host_statistics(_host_port, _HOST_CPU_LOAD_INFO, load, ctypes.byref(count)) != 0:
            return None
        user, system, idle, nice = load
        return (user, system, nice), (idle,), 2**32
    except Exception:
        return None


def host_busy(before: Optional[Ticks], after: Optional[Ticks]) -> Optional[float]:
    """The share of the host's CPU time that was busy between two readings."""
    if before is None or after is None:
        return None
    modulus = after[2]
    busy = sum((new - old) % modulus for new, old in zip(after[0], before[0]))
    idle = sum((new - old) % modulus for new, old in zip(after[1], before[1]))
    return None if busy + idle == 0 else busy / (busy + idle)


def yield_to_the_owner(pid: int) -> None:
    """Lower the command's CPU priority as it starts; its children inherit it.

    The gate keeps its own priority. A caller's higher niceness stays.
    """
    with contextlib.suppress(OSError):
        if os.getpriority(os.PRIO_PROCESS, pid) < RUN_NICENESS:
            os.setpriority(os.PRIO_PROCESS, pid, RUN_NICENESS)
