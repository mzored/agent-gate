"""agent-gate janitor: report what agents left running that nobody owns; with --apply, retire it.

    agent-gate janitor [--apply] [--json] [--simulator-hours H]

Without --apply it changes nothing. It retires only what is provably dead:

- orphan       a process launchd adopted (parent pid 1) whose working directory is gone:
               deleted, or replaced by another directory of the same name (a re-created worktree).
- log-stream   a `log stream` whose standard output is a pipe that no process can read. One that
               matches nothing never writes, so it never learns that its reader is gone.
- launchd-job  a launchd job of this user (not Apple's, not installed in /Library) whose working
               directory is gone: its plist's WorkingDirectory, or the directory its process runs in.
               launchd would keep restarting it.
- simulator    a simulator booted longer than --simulator-hours (default 2) with no live owner.

A directory is gone only when looking it up finds nothing there, or another directory; one that
cannot be read (privacy protection, a volume not mounted) is not gone.

Something with a live owner is always kept, and the report says why:

- a process whose session or process-group leader is still running, whose parent is (other than
  launchd), or that carries the run tag of a live gated run;
- a process, or a job's process, that may serve clients: it or a descendant has an established TCP
  connection on a port it listens on, listens on a Unix socket (whose clients cannot be told
  apart), or has descriptors that cannot be read;
- a simulator whose recorded owner (any process above the gated run that booted it) still runs,
  that a live process outside CoreSimulator names by its UDID, or any simulator while an app that
  shows or tests on simulators runs (SIMULATOR_APPS).

--apply reads the host again and retires only what both readings agree on. It stops processes with
TERM, then KILL, checking each identity (pid and start time) before every signal; boots out jobs with `launchctl bootout`, leaving any plist file in place; and shuts
simulators down with `simctl shutdown`, which keeps their data. Each retirement is logged to the
gate's event log. The report also shows Docker's disk image against the free disk.

Exit codes: 0, or 1 when --apply could not retire something; 78 when the host cannot be read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import errno
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import agent_gate
from gate_macos import (Process, Simulator, booted_simulators, identify, process_table, sockets, stdout_unread,
                        stop_processes, working_directory)

DEFAULT_SIMULATOR_HOURS = 2.0
LAUNCHCTL = "/bin/launchctl"
XCRUN = "/usr/bin/xcrun"
COMMAND_SECONDS = 30
#: Apps that show or test on simulators without naming them on their command line.
SIMULATOR_APPS = frozenset(("Simulator", "Xcode", "xcodebuild", "claude-ios-sim"))


@dataclass(frozen=True)
class Job:
    """A launchd job of this user: its label, the pid it runs as (None while stopped), where it came from
    (a plist path, or None for a submitted job) and the WorkingDirectory its plist names."""

    label: str
    pid: Optional[int]
    plist: Optional[str]
    working_directory: Optional[str]


@dataclass
class View:
    """One reading of the host: everything the rules decide on."""

    now: float
    processes: Dict[int, Process]
    #: pid -> the path of its working directory, for the processes a rule looks at.
    cwd: Dict[int, str] = field(default_factory=dict)
    #: pids whose working directory is gone.
    cwd_gone: FrozenSet[int] = frozenset()
    #: Jobs' WorkingDirectory paths that are gone.
    paths_gone: FrozenSet[str] = frozenset()
    #: pids whose standard output is a pipe no process reads.
    unread: FrozenSet[int] = frozenset()
    #: pid -> why clients may use it.
    serving: Dict[int, str] = field(default_factory=dict)
    jobs: List[Job] = field(default_factory=list)
    simulators: List[Simulator] = field(default_factory=list)
    #: UDIDs whose recorded owner still runs, with that owner's description.
    simulator_owners: Dict[str, str] = field(default_factory=dict)
    #: Run tags of live gated runs.
    live_runs: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class Finding:
    """A leftover: `retire` it, or keep it for `reason`, a live owner."""

    rule: str
    retire: bool
    subject: str
    evidence: str
    reason: str = ""
    pid: Optional[int] = None
    start: Optional[int] = None
    label: Optional[str] = None
    udid: Optional[str] = None

    @property
    def key(self) -> Tuple[object, ...]:
        return self.rule, self.pid, self.start, self.label, self.udid


# ------------------------------------------------------------------ rules


def descendants(view: View, pid: int) -> Set[int]:
    children: Dict[int, List[int]] = {}
    for process in view.processes.values():
        children.setdefault(process.ppid, []).append(process.pid)
    found, todo = set(), [pid]
    while todo:
        current = todo.pop()
        if current in found:
            continue
        found.add(current)
        todo.extend(children.get(current, ()))
    return found


def alive_before(view: View, pid: int, process: Process) -> Optional[Process]:
    """The live process `pid` when it predates `process`, so a recycled pid is never taken for it."""
    other = view.processes.get(pid)
    return other if other is not None and other.pid != process.pid and other.start <= process.start else None


def serving(view: View, pid: int) -> Optional[str]:
    """Why clients may use the process `pid` or a descendant; None when nothing says so."""
    return next((view.serving[member] for member in sorted(descendants(view, pid)) if member in view.serving), None)


def owner(view: View, process: Process) -> Optional[str]:
    """Why a live session or client still owns `process`; None when nothing does."""
    if process.ppid > 1:
        parent = alive_before(view, process.ppid, process)
        if parent is not None:
            return f"its parent {parent.name} (pid {parent.pid}) is running"
    for kind, pid in (("session", process.sid), ("process group", process.pgid)):
        leader = alive_before(view, pid, process) if pid > 1 else None
        if leader is not None:
            return f"its {kind} leader {leader.name} (pid {leader.pid}) is running"
    if process.run and process.run in view.live_runs:
        return f"it belongs to the live gated run {process.run}"
    return serving(view, process.pid)


def described(process: Process, view: View) -> str:
    hours = (view.now - process.start) / 3600
    argv = " ".join(process.argv) or process.name
    return f"pid {process.pid}, {hours:.1f}h old, pgid {process.pgid}, sid {process.sid}: {argv}"


def findings(view: View, simulator_hours: float = DEFAULT_SIMULATOR_HOURS) -> List[Finding]:
    """Every leftover the rules find in `view`: those to retire and those a live owner keeps."""
    found: List[Finding] = []
    job_pids = {job.pid for job in view.jobs if job.pid}

    for process in sorted(view.processes.values(), key=lambda process: process.pid):
        evidence = described(process, view)
        if process.name == "log" and process.argv[1:2] == ("stream",) and process.pid in view.unread:
            reason = owner(view, process)
            found.append(Finding("log-stream", reason is None, f"log stream (pid {process.pid})",
                                 f"its standard output is a pipe nobody reads; {evidence}", reason or "",
                                 process.pid, process.start))
        elif process.ppid == 1 and process.pid not in job_pids and process.pid in view.cwd_gone:
            reason = owner(view, process)
            found.append(Finding("orphan", reason is None, f"{process.name} (pid {process.pid})",
                                 f"its working directory {view.cwd.get(process.pid)} is gone; {evidence}",
                                 reason or "", process.pid, process.start))

    for job in sorted(view.jobs, key=lambda job: job.label):
        if job.label.startswith("com.apple.") or (job.plist or "").startswith(("/Library/", "/System/")):
            continue
        if job.pid in view.cwd_gone:
            path = view.cwd.get(job.pid)
        elif job.working_directory in view.paths_gone:
            path = job.working_directory
        else:
            continue
        reason = serving(view, job.pid) if job.pid else None
        origin = job.plist or "submitted with launchctl"
        state = f"pid {job.pid}" if job.pid else "not running"
        found.append(Finding("launchd-job", reason is None, job.label,
                             f"its working directory {path} is gone; {state}; {origin}", reason or "",
                             label=job.label))

    apps = sorted((process for process in view.processes.values() if process.name in SIMULATOR_APPS),
                  key=lambda process: process.pid)
    for simulator in view.simulators:
        hours = (view.now - simulator.booted) / 3600
        if hours < simulator_hours:
            continue
        reason = view.simulator_owners.get(simulator.udid)
        if reason is None:
            inside = descendants(view, simulator.pid) if simulator.pid else set()
            naming = [process for process in view.processes.values() if process.pid not in inside
                      and "CoreSimulator" not in " ".join(process.argv[:1])
                      and any(simulator.udid in argument.upper() for argument in process.argv)]
            if naming:
                reason = f"{naming[0].name} (pid {naming[0].pid}) uses it"
            elif apps:
                reason = f"{apps[0].name} (pid {apps[0].pid}) is running and may show or test on it"
        found.append(Finding("simulator", reason is None, f"simulator {simulator.name} ({simulator.udid})",
                             f"booted {hours:.1f}h ago", reason or "", udid=simulator.udid))
    return found


# ------------------------------------------------------------------ reading the host


def _run(command: Sequence[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=COMMAND_SECONDS,
                                stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return result.returncode, result.stdout if result.returncode == 0 else (result.stderr or result.stdout)


def directory_gone(path: str, inode: Optional[int] = None) -> bool:
    """Whether nothing is at `path` now, or another directory than the one with `inode`.

    A path that cannot be looked up for another reason, or that lies on a volume not mounted, is not gone.
    """
    try:
        now = os.stat(path).st_ino
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            return False
        parts = Path(path).parts
        return not (len(parts) > 2 and parts[1] == "Volumes" and not os.path.ismount(os.path.join(*parts[:3])))
    return inode is not None and now != inode


def _jobs() -> List[Job]:
    code, out = _run([LAUNCHCTL, "list"])
    if code != 0:
        raise agent_gate.ConfigError(f"launchctl list failed: {out.strip()}")
    jobs = []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) != 3 or parts[2].startswith("com.apple."):
            continue
        label = parts[2]
        code, printed = _run([LAUNCHCTL, "print", f"gui/{os.getuid()}/{label}"])
        if code != 0:
            continue  # not a job of this user's gui domain
        match = re.search(r"^\tpath = (/.*\.plist)$", printed, re.MULTILINE)
        plist = match.group(1) if match else None
        working = None
        if plist:
            try:
                with open(plist, "rb") as handle:
                    working = plistlib.load(handle).get("WorkingDirectory")
            except (OSError, ValueError, plistlib.InvalidFileException):
                pass
        jobs.append(Job(label, int(parts[0]) if parts[0].isdigit() else None, plist,
                        str(working) if working else None))
    return jobs


def _serving(view: View, roots: Sequence[int]) -> Dict[int, str]:
    """Why clients may use each process in the trees under `roots`."""
    reasons: Dict[int, str] = {}
    for pid in sorted(set().union(*(descendants(view, root) for root in roots)) if roots else ()):
        found = sockets(pid)
        if found is None:
            if identify([pid]).get(pid) == view.processes[pid].start:
                reasons[pid] = "its descriptors cannot be read"
            continue
        listening = {socket.port for socket in found if socket.listening and socket.port}
        clients = sorted({socket.port for socket in found if socket.established and socket.port in listening})
        if clients:
            reasons[pid] = f"a client is connected to its port {clients[0]}"
        elif any(socket.unix and socket.listening for socket in found):
            reasons[pid] = "it listens on a Unix socket, whose clients cannot be told apart"
    return reasons


def observe(settings: agent_gate.Settings) -> View:
    """Read the host once. ConfigError when the process table or launchd cannot be read."""
    table = process_table()
    if table is None:
        raise agent_gate.ConfigError("the process table is unreadable")
    view = View(now=time.time(), processes={process.pid: process for process in table})
    view.jobs = _jobs()
    job_pids = {job.pid for job in view.jobs if job.pid}
    roots = [process.pid for process in table if process.ppid == 1 or process.pid in job_pids]
    gone = set()
    for pid in roots:
        directory = working_directory(pid)
        if directory is not None:
            view.cwd[pid] = directory[0]
            if directory_gone(*directory):
                gone.add(pid)
    view.cwd_gone = frozenset(gone)
    view.paths_gone = frozenset(job.working_directory for job in view.jobs
                                if job.working_directory and directory_gone(job.working_directory))
    view.unread = frozenset(process.pid for process in table if process.name == "log" and stdout_unread(process.pid))
    view.serving = _serving(view, [pid for pid in roots if pid in gone or pid in view.unread])
    view.simulators = booted_simulators(settings.environ)
    for simulator in view.simulators:
        live = agent_gate.live_owner(settings, simulator)
        if live is not None:
            view.simulator_owners[simulator.udid] = (f"its owner {live.get('name')} (pid {live.get('pid')}) "
                                                     f"is running")
    view.live_runs = frozenset(str(job["id"]) for job in agent_gate.scan_jobs(settings))
    return view


# ------------------------------------------------------------------ retiring


def retire(settings: agent_gate.Settings, found: Sequence[Finding], hours: float) -> Dict[Tuple[object, ...], str]:
    """Retire each leftover in `found` that a second reading of the host still finds; why each other was not."""
    again = {finding.key for finding in findings(observe(settings), hours) if finding.retire}
    chosen = [finding for finding in found if finding.retire]
    failures = {finding.key: "it changed after the first reading; run the janitor again"
                for finding in chosen if finding.key not in again}
    chosen = [finding for finding in chosen if finding.key in again]
    targets = {finding.pid: finding.start for finding in chosen if finding.pid is not None and finding.start}

    def still_ours() -> Dict[int, int]:
        live = identify(list(targets))
        return {pid: start for pid, start in targets.items() if live.get(pid) == start}

    if targets:
        stop_processes(still_ours(), still_ours)
    survivors = still_ours() if targets else {}
    for finding in chosen:
        if finding.pid is not None:
            if finding.pid in survivors:
                failures[finding.key] = "it survived TERM and KILL"
        elif finding.label is not None:
            code, out = _run([LAUNCHCTL, "bootout", f"gui/{os.getuid()}/{finding.label}"])
            if code != 0:
                failures[finding.key] = f"launchctl bootout exited {code}: {out.strip()}"
        elif finding.udid is not None:
            code, out = _run([XCRUN, "simctl", "shutdown", finding.udid])
            if code != 0:
                failures[finding.key] = f"simctl shutdown exited {code}: {out.strip()}"
    for finding in found:
        if finding.retire:
            failure = failures.get(finding.key)
            agent_gate.log_event(settings, {"event": "retired" if failure is None else "retire-failed",
                                            "rule": finding.rule, "subject": finding.subject,
                                            "evidence": finding.evidence, "failure": failure})
    return failures


# ------------------------------------------------------------------ command


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="agent-gate janitor", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--apply", action="store_true", help="retire what the report lists to retire")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--simulator-hours", default=str(DEFAULT_SIMULATOR_HOURS), metavar="H",
                        help="how long a simulator with no live owner may stay booted")
    args = parser.parse_args(argv)
    try:
        settings = agent_gate.Settings()
        hours = agent_gate.finite(args.simulator_hours, "--simulator-hours")
        found = findings(observe(settings), hours)
        failures = retire(settings, found, hours) if args.apply and any(item.retire for item in found) else {}
    except (agent_gate.ConfigError, OSError) as error:
        return agent_gate.unusable(error)
    disk = agent_gate.docker_disk(settings)
    if args.json:
        print(json.dumps({
            "findings": [dict(finding.__dict__, failure=failures.get(finding.key)) for finding in found],
            "applied": args.apply,
            "docker_disk": disk,
            "disk_warnings": [] if disk is None else agent_gate.disk_warnings(settings, disk),
        }, indent=2, sort_keys=True))
        return 1 if failures else 0
    for finding in found:
        if finding.retire:
            state = "FAILED " if finding.key in failures else "retired" if args.apply else "retire "
        else:
            state = "keep   "
        print(f"{state} {finding.rule:<11} {finding.subject}: {finding.evidence}")
        if finding.reason:
            print(f"{'':20}kept: {finding.reason}")
        if finding.key in failures:
            print(f"{'':20}{failures[finding.key]}")
    if disk is not None:
        print("\n".join(agent_gate.disk_text(settings, disk)))
    pending = sum(1 for finding in found if finding.retire)
    if not found:
        print("no leftovers")
    elif pending and not args.apply:
        print(f"{pending} to retire: `agent-gate janitor --apply` retires them")
    return 1 if failures else 0
