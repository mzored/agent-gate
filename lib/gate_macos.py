"""macOS process inspection and machine signals used by agent-gate."""

from __future__ import annotations

import contextlib
import ctypes
import json
import math
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import time
from typing import Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

RUN_ENV = "AGENT_GATE_RUN"
# Tests can supply a file whose first line holds pressure, free-memory share and swap-out rate.
# It then stands in for the booted simulators too: each further line `simulator UDID BOOTED NAME`
# names one, BOOTED in epoch seconds, and no other counts.
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


def _bsd_info(pid: object) -> Optional[_BsdInfo]:
    """The kernel's record of a live process of this user; None when gone, a zombie or unreadable."""
    try:
        pid = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    info = _BsdInfo()
    if _proc_pidinfo(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return None
    return None if info.pbi_status == _SZOMB else info


def process_info(pid: object) -> Optional[Tuple[int, int]]:
    """(parent pid, start second) of a live process of this user; None when gone or a zombie.

    Other users' processes are unreadable and also yield None; every process the
    gate tracks belongs to this user.
    """
    info = _bsd_info(pid)
    return None if info is None else (int(info.pbi_ppid), int(info.pbi_start_tvsec))


def process_name(pid: object) -> str:
    """A live process's command name (at most 16 characters); "" when gone or unreadable."""
    info = _bsd_info(pid)
    return "" if info is None else info.pbi_comm.decode(errors="replace")


class Process(NamedTuple):
    """One process of this user, as the janitor sees it."""

    pid: int
    ppid: int
    pgid: int
    sid: int
    start: int
    name: str
    argv: Tuple[str, ...]
    #: The AGENT_GATE_RUN tag in its environment: the gated run it belongs to; "" when none.
    run: str = ""


def process_table() -> Optional[List[Process]]:
    """Every live process of this user, with its arguments where readable; None when the table is unreadable."""
    pids = all_pids()
    if pids is None:
        return None
    buffer = _args_buffer()
    table = []
    uid = os.getuid()
    for pid in pids:
        info = _bsd_info(pid)
        if info is None or info.pbi_uid != uid:
            continue
        try:
            sid = os.getsid(pid)
        except OSError:
            continue
        args = _procargs(pid, buffer)
        argv, environment = args if args else ([], [])
        tag = f"{RUN_ENV}="
        run = next((item[len(tag):] for item in environment if item.startswith(tag)), "")
        table.append(Process(pid, int(info.pbi_ppid), int(info.pbi_pgid), sid, int(info.pbi_start_tvsec),
                             info.pbi_comm.decode(errors="replace"), tuple(argv), run))
    return table


class _VinfoStat(ctypes.Structure):
    """struct vinfo_stat from <sys/proc_info.h>."""

    _fields_ = [("vst_dev", ctypes.c_uint32), ("vst_mode", ctypes.c_uint16), ("vst_nlink", ctypes.c_uint16),
                ("vst_ino", ctypes.c_uint64), ("vst_uid", ctypes.c_uint32), ("vst_gid", ctypes.c_uint32),
                ("vst_times", ctypes.c_int64 * 8), ("vst_size", ctypes.c_int64), ("vst_blocks", ctypes.c_int64),
                ("vst_blksize", ctypes.c_int32), ("vst_flags", ctypes.c_uint32), ("vst_gen", ctypes.c_uint32),
                ("vst_rdev", ctypes.c_uint32), ("vst_qspare", ctypes.c_int64 * 2)]


class _VnodeInfoPath(ctypes.Structure):
    """struct vnode_info_path."""

    _fields_ = [("vi_stat", _VinfoStat), ("vi_type", ctypes.c_int), ("vi_pad", ctypes.c_int),
                ("vi_fsid", ctypes.c_int32 * 2), ("vip_path", ctypes.c_char * 1024)]


class _VnodePathInfo(ctypes.Structure):
    """struct proc_vnodepathinfo: the working and root directories."""

    _fields_ = [("pvi_cdir", _VnodeInfoPath), ("pvi_rdir", _VnodeInfoPath)]


class _PipeFdInfo(ctypes.Structure):
    """struct pipe_fdinfo: struct proc_fileinfo, then struct pipe_info."""

    _fields_ = [("fi_openflags", ctypes.c_uint32), ("fi_status", ctypes.c_uint32), ("fi_offset", ctypes.c_int64),
                ("fi_type", ctypes.c_int32), ("fi_guardflags", ctypes.c_uint32), ("pipe_stat", _VinfoStat),
                ("pipe_handle", ctypes.c_uint64), ("pipe_peerhandle", ctypes.c_uint64), ("pipe_status", ctypes.c_int),
                ("rfu_1", ctypes.c_int)]


_PROC_PIDLISTFDS = 1
_PROC_PIDVNODEPATHINFO = 9
_PROC_PIDFDSOCKETINFO = 3
_PROC_PIDFDPIPEINFO = 6
_PROX_FDTYPE_SOCKET = 2
# Offsets into struct socket_fdinfo, which ctypes need not spell out whole.
_SOCKET_FDINFO_SIZE = 792
_SOI_FAMILY, _SOI_OPTIONS, _SOI_STATE, _SOI_KIND, _INSI_LPORT, _TCPSI_STATE = 184, 188, 192, 256, 268, 344
_AF_UNIX, _SO_ACCEPTCONN, _SS_ISCONNECTED, _SOCKINFO_TCP, _TSI_S_ESTABLISHED = 1, 0x2, 0x2, 2, 4
_proc_pidfdinfo = _LIBC.proc_pidfdinfo
_proc_pidfdinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_proc_pidfdinfo.restype = ctypes.c_int


def working_directory(pid: int) -> Optional[Tuple[str, int]]:
    """The path and inode of a live process's working directory; None when unreadable.

    The kernel keeps both for a directory deleted under the process: whether it is gone is what
    stat(2) of the path says now.
    """
    info = _VnodePathInfo()
    if _proc_pidinfo(pid, _PROC_PIDVNODEPATHINFO, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return None
    path = info.pvi_cdir.vip_path
    return (os.fsdecode(path), int(info.pvi_cdir.vi_stat.vst_ino)) if path else None


def stdout_unread(pid: int) -> bool:
    """Whether a process's standard output is a pipe whose read end no process holds."""
    info = _PipeFdInfo()
    if _proc_pidfdinfo(pid, 1, _PROC_PIDFDPIPEINFO, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        return False
    return info.pipe_peerhandle == 0


class Socket(NamedTuple):
    unix: bool
    listening: bool
    #: Its local TCP port; 0 for other sockets.
    port: int
    #: A TCP connection that is established.
    established: bool


def sockets(pid: int) -> Optional[List[Socket]]:
    """A live process's sockets; None when its descriptors are unreadable."""
    count = 1024
    while True:
        fds = (ctypes.c_int32 * (2 * count))()
        size = _proc_pidinfo(pid, _PROC_PIDLISTFDS, 0, fds, ctypes.sizeof(fds))
        if size <= 0:
            return None
        if size < ctypes.sizeof(fds):
            break
        count *= 4
    buffer = ctypes.create_string_buffer(_SOCKET_FDINFO_SIZE)
    found = []
    for index in range(size // 8):
        if fds[2 * index + 1] != _PROX_FDTYPE_SOCKET:
            continue
        if _proc_pidfdinfo(pid, fds[2 * index], _PROC_PIDFDSOCKETINFO, buffer, _SOCKET_FDINFO_SIZE) < _SOCKET_FDINFO_SIZE:
            continue  # closed meanwhile

        def field(offset: int, size: int = 4) -> int:
            return int.from_bytes(buffer.raw[offset:offset + size], sys.byteorder)

        tcp = field(_SOI_KIND) == _SOCKINFO_TCP
        port = int.from_bytes(field(_INSI_LPORT).to_bytes(4, sys.byteorder)[:2], "big") if tcp else 0
        found.append(Socket(field(_SOI_FAMILY) == _AF_UNIX, bool(field(_SOI_OPTIONS, 2) & _SO_ACCEPTCONN), port,
                            tcp and field(_TCPSI_STATE) == _TSI_S_ESTABLISHED))
    return found


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
        try:
            staging.write_text(json.dumps(readings))
            os.replace(staging, samples)
        except OSError:
            with contextlib.suppress(OSError):
                staging.unlink()
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
            words = (Path(override).read_text().splitlines() or [""])[0].split()[:3]
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


def memory_gib() -> Optional[float]:
    """Usable memory in GiB, the denominator of the free share; None when unreadable."""
    return _MEMORY_PAGES * _PAGE_BYTES / 2**30 if _MEMORY_PAGES else None


class Simulator(NamedTuple):
    """A booted simulator: its device, when it booted (its launchd_sim's start second) and that process."""

    udid: str
    name: str
    booted: float
    pid: int


#: launchd_sim's one argument names the device's bootstrap plist in its data directory.
_BOOTSTRAP_RE = re.compile(r"^(.*/([0-9A-Fa-f-]{36}))/data/var/run/launchd_bootstrap\.plist$")


def booted_simulators(environ: Mapping[str, str]) -> List[Simulator]:
    """The booted simulators, read from the process table: each runs one launchd_sim of this user.

    No simctl runs, so a waiter may read them on every poll. The host-signal
    override (MEMORY_FILE_ENV) stands in for them; an unreadable table reads as none.
    """
    override = environ.get(MEMORY_FILE_ENV)
    if override:
        found = []
        with contextlib.suppress(OSError):
            for line in Path(override).read_text().splitlines()[1:]:
                words = line.split(None, 3)
                if len(words) >= 3 and words[0] == "simulator":
                    with contextlib.suppress(ValueError):
                        found.append(Simulator(words[1], words[3] if len(words) > 3 else words[1], float(words[2]), 0))
        return found
    pids = all_pids() or []
    buffer: Optional["ctypes.Array[ctypes.c_char]"] = None
    found = []
    for pid in pids:
        info = _bsd_info(pid)
        if info is None or info.pbi_comm != b"launchd_sim":
            continue
        buffer = buffer or _args_buffer()
        args = _procargs(pid, buffer)
        match = _BOOTSTRAP_RE.match(args[0][-1]) if args and args[0] else None
        if match is None:
            continue
        name = match.group(2)
        with contextlib.suppress(OSError, ValueError, plistlib.InvalidFileException):
            with open(Path(match.group(1)) / "device.plist", "rb") as handle:
                name = str(plistlib.load(handle).get("name") or name)
        found.append(Simulator(match.group(2).upper(), name, float(info.pbi_start_tvsec), pid))
    return sorted(found, key=lambda simulator: simulator.booted)


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
