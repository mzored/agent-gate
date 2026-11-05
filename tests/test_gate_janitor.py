"""agent-gate janitor: each rule retires a dead leftover and keeps one a live owner still has.

The rules are decided on hand-made readings of the host. HostTest plants real leftovers and reads
the real host, which it never changes beyond its own processes. No test runs `--apply`.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import agent_gate  # noqa: E402
import gate_janitor  # noqa: E402
from gate_janitor import Job, View  # noqa: E402
from gate_macos import Process, Simulator, identify, process_info  # noqa: E402

NOW = 1_000_000.0
HOUR = 3600.0


def process(pid: int, ppid: int = 1, name: str = "node", argv: tuple = (), age: float = HOUR,
            pgid: Optional[int] = None, sid: Optional[int] = None, run: str = "") -> Process:
    return Process(pid, ppid, pgid or pid, sid or pid, int(NOW - age), name, argv or (name,), run)


def view(*processes: Process, **fields) -> View:
    return View(NOW, {item.pid: item for item in processes}, **fields)


class RuleTest(unittest.TestCase):
    def verdicts(self, reading: View, hours: float = 2.0) -> Dict[str, List[object]]:
        """rule -> [retire, reason] of each finding."""
        return {finding.rule: [finding.retire, finding.reason] for finding in gate_janitor.findings(reading, hours)}

    def test_an_orphan_whose_directory_is_gone_is_retired_unless_its_session_or_a_client_owns_it(self):
        vite = process(500, argv=("node", "vite"))
        gone = dict(cwd={500: "/w/app"}, cwd_gone=frozenset({500}))
        self.assertEqual(self.verdicts(view(vite, **gone)), {"orphan": [True, ""]})
        self.assertEqual(self.verdicts(view(vite, cwd={500: "/w/app"})), {})

        leader = process(400, ppid=300, name="zsh", age=2 * HOUR)
        kept = self.verdicts(view(vite._replace(sid=400), leader, **gone))
        self.assertEqual(kept["orphan"][0], False)
        self.assertIn("session leader zsh (pid 400)", kept["orphan"][1])
        # A pid recycled after the orphan started is not its leader.
        recycled = process(400, ppid=300, name="zsh", age=60)
        self.assertEqual(self.verdicts(view(vite._replace(sid=400), recycled, **gone)), {"orphan": [True, ""]})
        # Clients of it or of a process it started.
        child = process(501, ppid=500)
        self.assertEqual(self.verdicts(view(vite, child, **gone, serving={501: "a client is connected to its port 5173"})),
                         {"orphan": [False, "a client is connected to its port 5173"]})
        self.assertEqual(self.verdicts(view(vite._replace(run="r1"), **gone, live_runs=frozenset({"r1"})))["orphan"][0],
                         False)

    def test_a_log_stream_nobody_reads_is_retired_unless_its_parent_still_runs(self):
        stream = process(600, name="log", argv=("/usr/bin/log", "stream", "--predicate", "process == \"x\""))
        self.assertEqual(self.verdicts(view(stream, unread=frozenset({600}))), {"log-stream": [True, ""]})
        self.assertEqual(self.verdicts(view(stream)), {})

        shell = process(550, ppid=300, name="bash", age=2 * HOUR)
        kept = self.verdicts(view(stream._replace(ppid=550), shell, unread=frozenset({600})))
        self.assertEqual(kept["log-stream"][0], False)
        self.assertIn("parent bash (pid 550)", kept["log-stream"][1])

    def test_a_launchd_job_whose_directory_is_gone_is_retired_unless_a_client_is_connected(self):
        submitted = Job("com.example.vite.501.abc", 700, None, None)
        server = process(700)
        gone = dict(jobs=[submitted], cwd={700: "/w/client"}, cwd_gone=frozenset({700}))
        # Its process is the job's, not an orphan of its own.
        self.assertEqual(self.verdicts(view(server, **gone)), {"launchd-job": [True, ""]})
        child = process(701, ppid=700)
        self.assertEqual(self.verdicts(view(server, child, **gone, serving={701: "a client is connected to its port 80"})),
                         {"launchd-job": [False, "a client is connected to its port 80"]})

        stopped = Job("com.example.report", None, "/Users/me/Library/LaunchAgents/r.plist", "/w/gone")
        self.assertEqual(self.verdicts(view(jobs=[stopped], paths_gone=frozenset({"/w/gone"}))),
                         {"launchd-job": [True, ""]})
        apple = Job("com.apple.thing", None, None, "/w/gone")
        system = Job("org.vendor.agent", None, "/Library/LaunchAgents/a.plist", "/w/gone")
        self.assertEqual(self.verdicts(view(jobs=[apple, system], paths_gone=frozenset({"/w/gone"}))), {})

    def test_a_simulator_booted_for_hours_is_retired_unless_something_still_uses_it(self):
        udid = "760B9A8E-1FC9-4AF7-8FE7-02948899940A"
        old = Simulator(udid, "iPhone 17", NOW - 3 * HOUR, 800)
        self.assertEqual(self.verdicts(view(simulators=[old])), {"simulator": [True, ""]})
        self.assertEqual(self.verdicts(view(simulators=[old]), hours=4), {})
        self.assertEqual(self.verdicts(view(simulators=[old], simulator_owners={udid: "its owner claude is running"})),
                         {"simulator": [False, "its owner claude is running"]})
        for app in ("xcodebuild", "Simulator", "claude-ios-sim"):
            self.assertIn(f"{app} (pid 900)", self.verdicts(view(process(900, name=app), simulators=[old]))["simulator"][1])
        tool = process(950, ppid=850, name="node", argv=("node", "mcp", "--udid", udid.lower()))
        self.assertIn("node (pid 950) uses it", self.verdicts(view(tool, simulators=[old]))["simulator"][1])
        # The simulator's own processes name it too.
        inside = process(801, ppid=800, name="SpringBoard", argv=(f"/Devices/{udid}/SpringBoard",))
        self.assertEqual(self.verdicts(view(process(800, name="launchd_sim"), inside, simulators=[old])),
                         {"simulator": [True, ""]})


class HostTest(unittest.TestCase):
    """Real leftovers planted on this host, read by the real observer."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-gate-janitor-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        memory = self.tmp / "memory"
        memory.write_text("1 80")  # also hides the host's simulators
        self.settings = agent_gate.Settings({"HOME": str(self.tmp), "AGENT_GATE_DIR": str(self.tmp / "state"),
                                             "AGENT_GATE_MEMORY_FILE": str(memory)})

    def plant(self, script: str, new_session: bool = True) -> List[str]:
        """Run `script` in a grandchild launchd adopts; the words of the first line it prints, its pid first."""
        launcher = subprocess.Popen(
            [sys.executable, "-c", "import os, sys, time\nif os.fork(): os._exit(0)\n" + script],
            stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
            start_new_session=new_session)
        words = launcher.stdout.readline().split()  # type: ignore[union-attr]
        launcher.wait()
        pid = int(words[0])
        self.addCleanup(self.kill, pid, identify([pid]).get(pid))
        self.addCleanup(launcher.stdout.close)  # type: ignore[union-attr]
        deadline = time.monotonic() + 5
        while (process_info(pid) or (1,))[0] != 1 and time.monotonic() < deadline:
            time.sleep(0.02)
        return words

    def orphan(self, directory: Path, serve: str = "", new_session: bool = True) -> List[str]:
        """A process launchd adopted, working in `directory`; `serve` is code run after it prints its pid."""
        directory.mkdir(parents=True)
        return self.plant(f"os.chdir({str(directory)!r})\n{serve or 'print(os.getpid(), flush=True)'}\ntime.sleep(60)",
                          new_session)

    @staticmethod
    def kill(pid: int, start: Optional[int]) -> None:
        if start is not None and identify([pid]).get(pid) == start:
            os.kill(pid, signal.SIGKILL)

    def finding(self, pid: int) -> Optional[gate_janitor.Finding]:
        found = [item for item in gate_janitor.findings(gate_janitor.observe(self.settings)) if item.pid == pid]
        return found[0] if found else None

    def test_an_orphan_whose_directory_is_gone_is_reported_and_one_its_session_owns_is_kept(self):
        deleted = self.tmp / "deleted"
        alone = int(self.orphan(deleted)[0])
        os.rmdir(deleted)
        replaced = self.tmp / "replaced"
        again = int(self.orphan(replaced)[0])
        os.rmdir(replaced)
        replaced.mkdir()
        owned_directory = self.tmp / "owned"
        owned = int(self.orphan(owned_directory, new_session=False)[0])
        os.rmdir(owned_directory)

        found = self.finding(alone)
        self.assertEqual((found.rule, found.retire), ("orphan", True), found)
        self.assertEqual(self.finding(again).retire, True)
        kept = self.finding(owned)
        self.assertEqual((kept.rule, kept.retire), ("orphan", False), kept)
        self.assertIn("leader", kept.reason)

        self.assertEqual(gate_janitor.retire(self.settings, [found], 2.0), {})
        self.assertIsNone(process_info(alone))

    def test_a_directory_that_cannot_be_read_is_not_gone(self):
        hidden = self.tmp / "locked" / "work"
        pid = int(self.orphan(hidden)[0])
        hidden.parent.chmod(0)
        self.addCleanup(hidden.parent.chmod, 0o700)
        self.assertIsNone(self.finding(pid))

    def test_an_orphan_with_a_client_connected_or_a_unix_socket_is_kept(self):
        tcp = self.orphan(self.tmp / "tcp", serve=(
            "import socket\nserver = socket.socket()\nserver.bind(('127.0.0.1', 0))\nserver.listen()\n"
            "print(os.getpid(), server.getsockname()[1], flush=True)\nclient, _ = server.accept()"))
        os.rmdir(self.tmp / "tcp")
        self.assertEqual(self.finding(int(tcp[0])).retire, True, "no client yet")
        with socket.create_connection(("127.0.0.1", int(tcp[1]))):
            time.sleep(0.2)
            kept = self.finding(int(tcp[0]))
            self.assertEqual(kept.retire, False)
            self.assertEqual(kept.reason, f"a client is connected to its port {tcp[1]}")

        path = tempfile.mktemp(prefix="janitor-", dir="/tmp")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        unix = self.orphan(self.tmp / "unix", serve=(
            f"import socket\nserver = socket.socket(socket.AF_UNIX)\nserver.bind({path!r})\nserver.listen()\n"
            "print(os.getpid(), flush=True)"))
        os.rmdir(self.tmp / "unix")
        self.assertIn("Unix socket", self.finding(int(unix[0])).reason)

    def test_a_log_stream_that_lost_its_reader_is_reported(self):
        # The stream prints a header, then matches nothing, so it never writes again.
        words = self.plant(
            "os.setsid()\n"
            "read, write = os.pipe()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.dup2(write, 1)\n"
            "    os.close(read)\n"
            "    os.execv('/usr/bin/log', ['log', 'stream', '--predicate', 'process == \"agent-gate-none\"'])\n"
            "os.close(write)\n"
            "os.read(read, 4096)\n"
            "print(pid, flush=True)\n")
        found = self.finding(int(words[0]))
        self.assertEqual((found.rule, found.retire), ("log-stream", True), found)


if __name__ == "__main__":
    unittest.main()
