"""agent-gate on real processes. Every test uses its own state directory and pool."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence
import unittest
import unittest.mock

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "bin/agent-gate"
sys.path.insert(0, str(ROOT / "lib"))
import agent_gate  # noqa: E402
import gate_macos  # noqa: E402
import gate_host  # noqa: E402
import gate_policy  # noqa: E402

PY = sys.executable
# A shell starts background jobs with SIGINT ignored, and an ignored signal stays
# ignored across exec: give the gates under test the default a terminal gives.
if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    signal.signal(signal.SIGINT, signal.default_int_handler)

#: A command that marks its start and end. `release` is a file to wait for, or
#: "sleep:SECS" for a fixed duration.
HOLD = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path

    marks, name, release = sys.argv[1], sys.argv[2], sys.argv[3]

    def mark(*words):
        with open(marks, "a") as handle:
            handle.write(" ".join(map(str, words)) + "\\n")

    mark("start", name, time.time(), os.environ.get("AGENT_GATE_CPUS"), os.getpid())
    if release.startswith("sleep:"):
        time.sleep(float(release[6:]))
    else:
        while not Path(release).exists():
            time.sleep(0.02)
    mark("end", name, time.time())
    """
)

#: A `docker` that answers from files in $FAKE_DOCKER_DIR and records its calls with
#: its caller's pid: `vm` holds "AVAILABLE TOTAL" in GiB, `down` makes every call
#: fail, `hang` makes every call hang, `fail-<command>` fails that command, and
#: `images`, `ps` and `stats` hold what those commands print.
FAKE_DOCKER = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path

    state = Path(os.environ["FAKE_DOCKER_DIR"])
    args = sys.argv[1:]
    with open(state / "calls", "a") as handle:
        handle.write(" ".join([str(os.getppid()), *args]) + "\\n")

    def read(name, default=""):
        path = state / name
        return path.read_text() if path.exists() else default

    if (state / "down").exists():
        sys.exit("Cannot connect to the Docker daemon")
    if (state / "hang").exists():
        import time
        time.sleep(60)
    if (state / f"fail-{args[0]}").exists():
        sys.exit(f"docker {args[0]} failed")
    if args[:2] == ["image", "ls"]:
        print(read("images", "alpine:latest"))
    elif args[0] == "run":
        available, total = (float(word) * 2**20 for word in read("vm").split())
        print(f"MemTotal:       {total:.0f} kB\\nMemFree:        1 kB\\nMemAvailable:   {available:.0f} kB")
    elif args[0] in ("ps", "stats"):
        print(read(args[0]), end="")
    else:
        sys.exit(f"unexpected docker {args}")
    """
)


def command_of(pid: int) -> str:
    """The command line of one of this user's processes; "" when unreadable.

    macOS returns no environment for platform binaries such as sh and sleep.
    """
    args = gate_macos._procargs(pid, gate_macos._args_buffer())
    return " ".join(args[0]) if args else ""


class Result(NamedTuple):
    code: int
    out: str
    err: str
    elapsed: float


class GateCase(unittest.TestCase):
    tokens = 4

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agent-gate-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.procs: List[subprocess.Popen] = []
        self.leftovers: List[int] = []
        self.addCleanup(self._stop_everything)
        self.state = self.tmp / "state"
        self.memory = self.tmp / "memory"
        self.set_memory("1 80")
        self.marks = self.tmp / "marks"
        self.hold = self.tmp / "hold.py"
        self.hold.write_text(HOLD)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AGENT_", "VITEST_", "PYTEST_XDIST", "DEVSHIM_"))
        }
        self.env.update(
            AGENT_GATE_DIR=str(self.state),
            AGENT_GATE_MEMORY_FILE=str(self.memory),
            AGENT_GATE_TOKENS=str(self.tokens),
            # Most tests exercise claims up to the whole pool; ClaimTest covers the per-run cap.
            AGENT_GATE_MAX_CLAIM=str(self.tokens),
            AGENT_GATE_RAMP_SECONDS="0",
            AGENT_GATE_MIN_FREE_PCT="25",
            # Strict FIFO without headroom, as most tests assume; BackfillTest enables both.
            AGENT_GATE_BACKFILL_SECONDS="0",
            AGENT_GATE_SMALL_HEADROOM="0",
        )

    def _stop_everything(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                with self.suppress():
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for pid in self.leftovers:
            if gate_macos.process_info(pid) is not None:
                with self.suppress():
                    os.kill(pid, signal.SIGKILL)

    @staticmethod
    def suppress():
        import contextlib

        return contextlib.suppress(ProcessLookupError, PermissionError)

    # ---------------------------------------------------------------- helpers

    def set_memory(self, text: str) -> None:
        staging = self.memory.with_name("memory.tmp")
        staging.write_text(text)
        os.replace(staging, self.memory)

    def held(self, name: str, release: Optional[str] = None) -> List[str]:
        return [PY, str(self.hold), str(self.marks), name, release or str(self.tmp / f"release-{name}")]

    def release(self, name: str) -> None:
        (self.tmp / f"release-{name}").touch()

    def holding(self, name: str, k: int, **kwargs) -> subprocess.Popen:
        """Start a gate claiming `k` tokens around a command that runs until `end`, and wait for it to start."""
        proc = self.start("-c", k, "--", *self.held(name), name=name, **kwargs)
        self.command_pid(name)
        return proc

    def end(self, *procs: subprocess.Popen) -> None:
        """Let the commands of gates started by `holding` finish, and expect each gate to exit 0."""
        for proc in procs:
            self.release(proc.name)  # type: ignore[attr-defined]
        for proc in procs:
            self.assertEqual(self.finish(proc).code, 0, self.err(proc.name))  # type: ignore[attr-defined]

    def sleeper(self, code: str = "import time; time.sleep(60)", **kwargs) -> subprocess.Popen:
        """A background process that is killed and reaped when the test ends."""
        proc = subprocess.Popen([PY, "-c", code], **kwargs)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        return proc

    @staticmethod
    def gate_script(body: str) -> str:
        """A script for a command that runs inside a gate; `gate` is the agent-gate command line to extend."""
        return f"import os, subprocess, sys\ngate = [sys.executable, {str(GATE)!r}]\n" + textwrap.dedent(body)

    def start(self, *args: object, env: Optional[Dict[str, str]] = None, name: str = "gate",
              program: Path = GATE, cwd: Optional[Path] = None) -> subprocess.Popen:
        """Start a gate in the background. Its directory is the test's own, so that the gate's fingerprint
        of the working tree does not read the repository; a test that needs the repository passes `cwd`."""
        out = open(self.tmp / f"{name}.out", "w")
        err = open(self.tmp / f"{name}.err", "w")
        self.addCleanup(out.close)
        self.addCleanup(err.close)
        proc = subprocess.Popen(
            [PY, str(program), *map(str, args)],
            env=self.env if env is None else env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            cwd=self.tmp if cwd is None else cwd,
            start_new_session=True,
        )
        proc.name = name  # type: ignore[attr-defined]
        proc.began = time.monotonic()  # type: ignore[attr-defined]
        self.procs.append(proc)
        return proc

    def finish(self, proc: subprocess.Popen, timeout: float = 30) -> Result:
        try:
            code = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.fail(f"{proc.name} still running after {timeout}s: {self.err(proc.name)}")  # type: ignore[attr-defined]
        name = proc.name  # type: ignore[attr-defined]
        return Result(code, self.out(name), self.err(name), time.monotonic() - proc.began)  # type: ignore[attr-defined]

    def gate(self, *args: object, timeout: float = 30, **kwargs) -> Result:
        kwargs.setdefault("name", f"run{len(self.procs)}")
        return self.finish(self.start(*args, **kwargs), timeout)

    def out(self, name: str) -> str:
        return (self.tmp / f"{name}.out").read_text()

    def err(self, name: str) -> str:
        return (self.tmp / f"{name}.err").read_text()

    def wait_for(self, predicate: Callable[[], object], what: str, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail(f"timed out waiting for {what}")
            time.sleep(0.05)

    def mark_lines(self) -> List[List[str]]:
        try:
            return [line.split() for line in self.marks.read_text().splitlines() if line.strip()]
        except FileNotFoundError:
            return []

    def times(self, kind: str) -> Dict[str, float]:
        return {words[1]: float(words[2]) for words in self.mark_lines() if words[0] == kind}

    def started(self, name: str) -> bool:
        return name in self.times("start")

    def command_pid(self, name: str) -> int:
        self.wait_for(lambda: self.started(name), f"{name} to start")
        pid = next(int(words[4]) for words in self.mark_lines() if words[0] == "start" and words[1] == name)
        self.leftovers.append(pid)
        return pid

    def settings(self) -> agent_gate.Settings:
        return agent_gate.Settings(self.env)

    def queued(self, docker: bool = False) -> int:
        """Live requests in the token queue, or in the Docker queue."""
        settings = self.settings()
        directory = settings.docker_waiters if docker else settings.waiters
        return len(agent_gate.live_waiters(settings, collect=False, directory=directory))

    def ensure_state(self) -> None:
        """Create the state directory as a first run does, for tests that write into it themselves."""
        self.settings().ensure()

    def jobs(self) -> List[Dict[str, object]]:
        return agent_gate.scan_jobs(self.settings())

    def events(self, kind: str) -> List[Dict[str, object]]:
        try:
            lines = (self.state / "events.jsonl").read_text().splitlines()
        except FileNotFoundError:
            return []
        return [event for event in map(json.loads, lines) if event.get("event") == kind]

    def status(self) -> Dict[str, object]:
        result = subprocess.run([PY, str(GATE), "status", "--json"], env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def status_text(self) -> str:
        return subprocess.run([PY, str(GATE), "status"], env=self.env, capture_output=True, text=True).stdout

    def gone(self, pid: int) -> bool:
        return gate_macos.process_info(pid) is None


class PolicyTest(unittest.TestCase):
    """The admission rule and the fan-out, decided on a snapshot of the live runs and the kernel's memory signal."""

    NOW = 1_000_000.0

    def decision(self, used: float, pressure: Optional[int], free: Optional[int], since: float = 60.0, k: int = 2,
                 ahead: Sequence[Dict[str, object]] = (), outside: Optional[float] = None,
                 swapout: Optional[float] = 0.0, grow: bool = False, workers: Optional[float] = None,
                 **environ: str) -> gate_policy.Decision:
        """`used` tokens held by live runs, `outside` of them not in the headroom (all of `used` up to T by
        default), and `workers` the CPUs their runners use (their tokens by default)."""
        settings = agent_gate.Settings({"AGENT_GATE_TOKENS": "10", "AGENT_GATE_MIN_FREE_PCT": "25", **environ})
        outside = min(used, settings.tokens) if outside is None else outside
        jobs: List[Dict[str, object]] = []
        if outside:
            jobs.append({"k": outside, "workers": outside if workers is None else workers})
        if used - outside:
            jobs.append({"k": used - outside, "headroom": True})
        snapshot = gate_policy.Snapshot(jobs, pressure, free, swapout, since)
        return gate_policy.decide(snapshot, k, settings, ahead, self.NOW, grow)

    def decide(self, *args, **kwargs) -> str:
        return self.decision(*args, **kwargs).code

    def waiter(self, k: int, waited: float = 1.0) -> Dict[str, object]:
        """A request queued ahead."""
        return {"k": k, "queued": self.NOW - waited}

    def test_one_run_always_starts_on_an_idle_machine_unless_pressure_is_critical(self):
        self.assertEqual(self.decide(0, 2, 5), "start")
        self.assertEqual(self.decide(0, None, None), "start")
        self.assertEqual(self.decide(0, 4, 80), "critical")

    def test_warning_pressure_alone_does_not_throttle(self):
        self.assertEqual(self.decide(8, 2, 25), "start")

    def test_tokens_free_memory_and_ramp_limit_concurrent_runs(self):
        self.assertEqual(self.decide(9, 1, 80), "tokens")
        self.assertEqual(self.decide(2, 1, 24), "free")
        self.assertEqual(self.decide(2, 1, 29, AGENT_GATE_MIN_FREE_PCT="30"), "free")
        self.assertEqual(self.decide(2, 1, 30, since=1.0), "ramp")
        self.assertEqual(self.decide(2, 1, 30, since=1.0, AGENT_GATE_RAMP_SECONDS="0"), "start")
        self.assertEqual(self.decide(1, None, None), "unreadable")
        self.assertEqual(self.decide(1, 4, 80), "critical")
        self.assertEqual(self.decide(1, 1, 80, swapout=None), "measuring")
        self.assertEqual(self.decide(1, 1, 80, swapout=50), "swap")
        self.assertEqual(self.decide(0, 1, 80, swapout=None), "start", "an idle host starts one run without measuring")

    def test_a_one_token_run_beside_others_is_held_only_by_measured_harm(self):
        self.assertEqual(self.decide(2, 1, 10, k=1), "start", "low free memory is a prediction")
        self.assertEqual(self.decide(2, 1, 30, since=0.0, k=1), "start", "the ramp spaces larger claims")
        self.assertEqual(self.decide(2, 1, 80, swapout=None, k=1), "start", "no rate measured is no harm")
        self.assertEqual(self.decide(2, 1, 80, swapout=50, k=1), "swap")
        self.assertEqual(self.decide(2, 4, 80, k=1), "critical")
        self.assertEqual(self.decide(2, None, None, k=1), "unreadable")

    def test_the_ramp_spaces_admissions_only_while_free_memory_is_thin(self):
        # Free memory within 10 points of the 25% minimum, or unreadable, is thin.
        self.assertEqual(self.decide(2, 1, 34, since=1.0), "ramp")
        self.assertEqual(self.decide(2, 1, 35, since=1.0), "start")
        self.assertEqual(self.decide(2, 1, 80, since=0.0), "start")
        self.assertEqual(self.decide(2, 1, 30, since=1.0, AGENT_GATE_RAMP_MARGIN_PCT="0"), "start")
        self.assertEqual(self.decide(2, 1, 50, since=1.0, AGENT_GATE_RAMP_MARGIN_PCT="30"), "ramp")

    def test_one_token_runs_have_headroom_beyond_the_pool(self):
        self.assertEqual(self.decide(10, 1, 80, k=1), "start")
        self.assertTrue(self.decision(10, 1, 80, k=1).headroom)
        self.assertEqual(self.decide(11, 1, 80, k=1), "start")
        self.assertEqual(self.decision(12, 1, 80, k=1).reason, "12 of 10 tokens in use (+2 for 1-token runs), 1 needed")
        self.assertEqual(self.decide(10, 1, 80, k=2), "tokens")
        self.assertEqual(self.decide(9, 1, 80, k=2), "tokens", "small runs in the headroom count for larger claims")
        self.assertEqual(self.decide(10, 1, 80, k=1, AGENT_GATE_SMALL_HEADROOM="0"), "tokens")
        for tokens, headroom in (("1", 1), ("4", 1), ("10", 2), ("16", 3)):
            self.assertEqual(agent_gate.Settings({"AGENT_GATE_TOKENS": tokens}).headroom, headroom)

    def test_a_later_request_passes_only_requests_that_wait_for_tokens_alone(self):
        head = [self.waiter(5)]
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=head), "start")
        self.assertTrue(self.decision(8, 1, 80, k=1, ahead=head).backfilled)
        self.assertEqual(self.decide(8, 1, 80, k=2, ahead=head), "start")
        self.assertEqual(self.decide(8, 1, 80, k=3, ahead=head), "tokens")
        self.assertEqual(self.decide(5, 1, 80, k=1, ahead=head), "fits-first")
        self.assertEqual(self.decide(0, 1, 80, k=1, ahead=head), "fits-first")
        self.assertEqual(self.decide(11, 1, 80, k=1, ahead=head + [self.waiter(1)]), "fits-first",
                         "a 1-token request ahead that fits its headroom goes first")
        # Memory and the ramp hold the head and the later request alike.
        self.assertEqual(self.decide(8, 1, 10, k=2, ahead=head), "free")
        self.assertEqual(self.decide(8, 1, 30, since=1.0, k=2, ahead=head), "ramp")
        self.assertEqual(self.decide(8, 4, 80, k=1, ahead=head), "critical")

    def test_only_a_one_token_run_into_a_full_pools_headroom_passes_an_aged_request(self):
        aged = [self.waiter(5, waited=60)]
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=aged), "aged")
        self.assertEqual(self.decide(9, 1, 80, k=1, ahead=aged), "aged")
        # Runs outside the headroom hold all T tokens: the aged request cannot start before one of
        # them ends, so a 1-token run may take the headroom.
        self.assertEqual(self.decide(10, 1, 80, k=1, ahead=aged), "start")
        self.assertEqual(self.decide(11, 1, 80, k=1, ahead=aged), "start")
        # Headroom runs fill the pool: once a run outside the headroom has ended, nothing passes.
        self.assertEqual(self.decide(10, 1, 80, k=1, ahead=aged, outside=9), "aged")
        self.assertEqual(self.decide(11, 1, 80, k=1, ahead=aged, outside=9), "aged")
        self.assertEqual(self.decide(12, 1, 80, k=1, ahead=aged), "aged", "the headroom is full")
        self.assertEqual(self.decide(10, 1, 80, k=1, ahead=aged, AGENT_GATE_SMALL_HEADROOM="0"), "aged")
        self.assertEqual(self.decide(10, 1, 80, k=2, ahead=aged), "aged")
        self.assertEqual(self.decide(10, 1, 80, k=1, ahead=[self.waiter(1, waited=60)]), "fits-first")
        # Measured harm still holds it.
        self.assertEqual(self.decide(10, 1, 80, swapout=50, k=1, ahead=aged), "swap")
        self.assertEqual(self.decide(10, 4, 80, k=1, ahead=aged), "critical")

    def test_nothing_passes_an_aged_or_malformed_request_while_tokens_are_free(self):
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=[self.waiter(5, waited=60)]), "aged")
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=[self.waiter(1), self.waiter(5, 61)]), "aged")
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=[self.waiter(5, waited=60)],
                                     AGENT_GATE_BACKFILL_SECONDS="61"), "start")
        self.assertEqual(self.decide(8, 1, 80, k=1, ahead=[self.waiter(5, waited=0)],
                                     AGENT_GATE_BACKFILL_SECONDS="0"), "aged")
        for record in ({**self.waiter(5), "k": None}, {}):
            with self.subTest(record=record):
                self.assertEqual(self.decide(8, 1, 80, k=1, ahead=[record]), "malformed")

    def test_a_default_claim_gets_the_cpus_no_other_run_uses(self):
        self.assertEqual(self.decision(0, 1, 80, grow=True).workers, 10, "alone: the whole machine")
        self.assertEqual(self.decision(0, 1, 80).workers, 2, "an explicit claim keeps its size")
        self.assertEqual(self.decision(2, 1, 80, grow=True, workers=7).workers, 3)
        self.assertEqual(self.decision(2, 1, 80, grow=True, workers=10).workers, 2, "never below its claim")
        self.assertEqual(self.decision(0, 1, 80, grow=True, swapout=None).workers, 10, "no rate measured is no harm")
        for memory in ({"free": 10}, {"swapout": 50}, {"pressure": 4}, {"free": None}):
            with self.subTest(memory=memory):
                snapshot = gate_policy.Snapshot([], **{"pressure": 1, "free": 80, "swapout": 0.0, **memory},
                                                since_admit=60.0)
                self.assertEqual(gate_policy.fan_out(snapshot, 2, True, agent_gate.Settings({"AGENT_GATE_TOKENS": "10",
                                 "AGENT_GATE_MIN_FREE_PCT": "25"})), 2, "memory without room keeps the claim")

    def test_an_unreadable_pressure_level_leaves_the_free_memory_rule(self):
        # Inside the Codex sandbox only the free share is readable.
        self.assertEqual(self.decide(2, None, 80), "start")
        self.assertEqual(self.decide(2, None, 10), "free")

    def test_the_free_share_leaves_out_app_memory_the_compressor_holds(self):
        # 60 truly free (free_count includes the 40 speculative), 150 file-backed (the 40
        # speculative among them) and 30 purgeable pages are free: 240 of 1000. Active and
        # inactive app memory, wired and compressed pages are not, however much of them there is.
        stats = gate_macos._VmStatistics64(
            free_count=100, speculative_count=40, external_page_count=150, purgeable_count=30,
            active_count=300, inactive_count=200, internal_page_count=350, wire_count=300,
            throttled_count=50, compressor_page_count=200,
        )
        self.assertEqual(gate_macos.free_share(stats, 1000), 24)
        self.assertEqual(self.decide(2, 1, gate_macos.free_share(stats, 1000)), "free")
        self.assertIsNone(gate_macos.free_share(stats, None))

    def test_tuning_variables_must_be_finite(self):
        for name in ("AGENT_GATE_MIN_FREE_PCT", "AGENT_GATE_RAMP_SECONDS", "AGENT_GATE_RAMP_MARGIN_PCT",
                     "AGENT_GATE_BACKFILL_SECONDS"):
            for text in ("nan", "inf", "-1", "x"):
                with self.subTest(name=name, text=text), self.assertRaises(agent_gate.ConfigError):
                    agent_gate.Settings({name: text})
        for text in ("-1", "1.5", "x"):
            with self.subTest(name="AGENT_GATE_SMALL_HEADROOM", text=text), self.assertRaises(agent_gate.ConfigError):
                agent_gate.Settings({"AGENT_GATE_SMALL_HEADROOM": text})
        self.assertEqual(agent_gate.Settings({"AGENT_GATE_SMALL_HEADROOM": "0"}).headroom, 0)


class CapacityTest(GateCase):
    def test_running_tokens_never_exceed_the_pool(self):
        rng = random.Random(7)
        jobs = [(f"j{index}", rng.choice((1, 1, 2, 3, 4)), rng.uniform(0.1, 0.4)) for index in range(16)]
        procs = [self.start("-c", k, "--", *self.held(name, f"sleep:{seconds:.2f}"), name=name)
                 for name, k, seconds in jobs]
        for proc in procs:
            self.assertEqual(self.finish(proc, 90).code, 0, self.err(proc.name))  # type: ignore[attr-defined]
        tokens = {name: k for name, k, _ in jobs}
        cpus = {words[1]: words[3] for words in self.mark_lines() if words[0] == "start"}
        self.assertEqual(cpus, {name: str(k) for name, k in tokens.items()})
        timeline = [(at, 1, tokens[name]) for name, at in self.times("start").items()]
        timeline += [(at, 0, -tokens[name]) for name, at in self.times("end").items()]
        used = peak = running = most = 0
        for _, _, delta in sorted(timeline):
            used += delta
            running += 1 if delta > 0 else -1
            peak, most = max(peak, used), max(most, running)
        self.assertLessEqual(peak, self.tokens)
        self.assertGreater(most, 1, "runs that fit together should overlap")

    def test_requests_outside_the_pool_are_configuration_errors(self):
        for args in (("-c", "0"), ("-c", "-1"), ("-c", "nan"), ("-c", "inf"), ("--timeout", "nan"),
                     ("--timeout", "-1"), ("--timeout", "inf"), ("--wait", "nan"), ("--wait", "-1")):
            with self.subTest(args=args):
                result = self.gate(*args, "--", *self.held("never", "sleep:0"))
                self.assertEqual(result.code, agent_gate.CONFIG_EXIT, result.err)
                self.assertIn("unusable", result.err)
        self.assertFalse(self.started("never"))
        result = self.gate("-c", "1.5", "--", *self.held("rounded", "sleep:0"))
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(next(w[3] for w in self.mark_lines() if w[1] == "rounded"), "2")

    def test_the_command_exit_code_passes_through(self):
        self.assertEqual(self.gate("--", PY, "-c", "raise SystemExit(3)").code, 3)
        self.assertEqual(self.gate("--", str(self.tmp / "missing")).code, 127)


class ClaimTest(GateCase):
    """No single run holds more than the per-run cap, half the tokens unless configured."""

    PROBE = "import os; print(os.environ['AGENT_GATE_CPUS'], os.environ['PYTEST_XDIST_AUTO_NUM_WORKERS'])"

    def setUp(self) -> None:
        super().setUp()
        del self.env["AGENT_GATE_MAX_CLAIM"]

    def test_a_larger_claim_runs_with_the_cap_and_its_runners_follow(self):
        # A default claim reserves 2 and, alone, its runners get every CPU.
        for args, granted, told in ((("-c", 4), "2", True), (("-c", 9), "2", True), ((), "4", False),
                                    (("-c", 1), "1", False)):
            with self.subTest(args=args):
                result = self.gate(*args, "--", PY, "-c", self.PROBE)
                self.assertEqual(result.code, 0, result.err)
                self.assertEqual(result.out.split(), [granted, granted])
                self.assertEqual("runs with 2" in result.err, told, result.err)
        released = self.events("released")[0]
        self.assertEqual((released["k"], released["asked"]), (2, 4))

    def test_the_cap_is_configurable_within_the_tokens(self):
        for extra, asked, granted in (({"AGENT_GATE_MAX_CLAIM": "3"}, 4, "3"), ({"AGENT_GATE_MAX_CLAIM": "9"}, 9, "4"),
                                      ({"AGENT_GATE_TOKENS": "5"}, 5, "2"), ({"AGENT_GATE_TOKENS": "1"}, 1, "1")):
            with self.subTest(extra=extra):
                result = self.gate("-c", asked, "--", PY, "-c", self.PROBE, env=dict(self.env, **extra))
                self.assertEqual(result.code, 0, result.err)
                self.assertEqual(result.out.split(), [granted, granted])
        result = self.gate("--", PY, "-c", "pass", env=dict(self.env, AGENT_GATE_MAX_CLAIM="0"))
        self.assertEqual(result.code, agent_gate.CONFIG_EXIT, result.err)

    def test_two_whole_machine_claims_run_side_by_side(self):
        first = self.start("-c", 10, "--", *self.held("first"), name="first")
        second = self.start("-c", 10, "--", *self.held("second"), name="second")
        self.wait_for(lambda: self.started("first") and self.started("second"), "both capped runs to start")
        for name, proc in (("first", first), ("second", second)):
            self.release(name)
            self.assertEqual(self.finish(proc).code, 0)


class FanOutTest(GateCase):
    """A default claim's runners get the CPUs no other run's runners use; an explicit -c keeps its size."""

    PROBE = "import os; print(os.environ['AGENT_GATE_CPUS'], os.environ['VITEST_MAX_WORKERS'])"

    def workers(self, *args: object) -> List[str]:
        result = self.gate(*args, "--", PY, "-c", self.PROBE)
        self.assertEqual(result.code, 0, result.err)
        return result.out.split()

    def test_a_lone_default_run_uses_the_whole_machine(self):
        self.assertEqual(self.workers(), ["4", "4"])
        self.assertEqual(self.workers("-c", 2), ["2", "2"])
        self.assertEqual(self.events("released")[0]["workers"], 4)

    def test_beside_other_runs_a_default_run_gets_what_they_leave(self):
        self.env["AGENT_GATE_TOKENS"] = "6"
        small = self.holding("small", 1)
        self.assertEqual(self.workers(), ["5", "5"])
        wide = self.start("--", *self.held("wide"), name="wide")
        self.command_pid("wide")
        self.assertEqual(self.workers(), ["2", "2"], "never below the claim it reserves")
        self.end(small, wide)

    def test_without_memory_to_spare_a_default_run_keeps_its_claim(self):
        self.set_memory("1 10")
        self.assertEqual(self.workers(), ["2", "2"])

    def test_a_nested_gate_and_inside_get_the_runs_workers(self):
        script = self.gate_script(
            """
            subprocess.run(gate + ["inside", "-c", "3"], check=True)
            subprocess.run(gate + ["--", sys.executable, "-c", "import os; print(os.environ['AGENT_GATE_CPUS'])"],
                           check=True)
            """
        )
        result = self.gate("--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.split(), ["4"])


class SwapTest(GateCase):
    """The swap-out rate is shared by every gate on the host, so a gate that just started need not measure it."""

    def test_a_new_gate_beside_a_running_one_starts_without_measuring(self):
        env = {key: value for key, value in self.env.items() if key != "AGENT_GATE_MEMORY_FILE"}
        # The real kernel signal, with limits no host reaches: only an unmeasured rate could hold it.
        env.update(AGENT_GATE_MIN_FREE_PCT="0", AGENT_GATE_MAX_SWAPOUT_MBPS="1000000")
        blocker = self.start("-c", 1, "--", *self.held("blocker"), env=env, name="blocker")
        self.command_pid("blocker")
        time.sleep(1.2)
        result = self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass", env=env)
        self.assertEqual(result.code, 0, result.err)
        self.assertIsNotNone(self.events("admitted")[-1]["swapout"])
        self.release("blocker")
        self.assertEqual(self.finish(blocker).code, 0)


class QueueTest(GateCase):
    def test_the_queue_is_strict_fifo_and_reports_positions(self):
        blocker = self.holding("blocker", 4)
        a = self.start("-c", 1, "--", *self.held("a", "sleep:0.8"), name="a")
        self.wait_for(lambda: self.queued() == 1, "a to queue")
        b = self.start("-c", 4, "--", *self.held("b", "sleep:0.1"), name="b")
        self.wait_for(lambda: self.queued() == 2, "b to queue")
        c = self.start("-c", 1, "--", *self.held("c", "sleep:0.1"), name="c")
        self.wait_for(lambda: self.queued() == 3, "c to queue")
        self.wait_for(lambda: "position 3" in self.err("c"), "c to report its position")
        self.assertIn("position 2", self.err("b"))
        self.assertIn("4 of 4 tokens in use", self.err("a"))
        self.assertIn("Waiting is not a failure", self.err("c"))
        text = self.status_text()
        self.assertIn("running k=4", text)
        self.assertIn("queued #3 k=1", text)
        self.release("blocker")
        for proc in (blocker, a, b, c):
            self.assertEqual(self.finish(proc).code, 0)
        starts, ends = self.times("start"), self.times("end")
        self.assertLess(starts["a"], starts["b"])
        self.assertGreaterEqual(starts["b"], ends["a"], "b needs every token, so it waits for a")
        self.assertGreaterEqual(starts["c"], ends["b"], "c fits beside a but must not overtake b")
        self.assertEqual(os.listdir(self.state / "waiters"), [])
        self.assertEqual(os.listdir(self.state / "jobs"), [])
        self.assertEqual(self.events("abandoned"), [], "a waiter that took its turn did not abandon it")

    def test_a_dead_waiter_leaves_the_queue(self):
        blocker = self.holding("blocker", 4)
        a = self.start("-c", 1, "-l", "a", "--", *self.held("a", "sleep:0"), name="a")
        self.wait_for(lambda: self.queued() == 1, "a to queue")
        b = self.start("-c", 1, "--", *self.held("b", "sleep:0"), name="b")
        self.wait_for(lambda: self.queued() == 2, "b to queue")
        a.kill()
        a.wait()
        self.wait_for(lambda: self.queued() == 1, "the kernel to drop a's lock")
        # b, right behind, wakes as the lock goes and logs a's request, once.
        self.wait_for(lambda: self.events("abandoned"), "a's request in the log")
        self.release("blocker")
        self.assertEqual(self.finish(b).code, 0)
        self.assertEqual(self.finish(blocker).code, 0)
        self.assertFalse(self.started("a"))
        self.assertEqual(os.listdir(self.state / "waiters"), [])
        [event] = self.events("abandoned")
        self.assertEqual((event["queue"], event["label"], event["k"], event["pid"]), ("tokens", "a", 1, a.pid))
        self.assertLess(event["waited_at_most_s"], time.monotonic() - a.began)  # type: ignore[attr-defined]

    def test_an_interrupted_waiter_leaves_on_its_own(self):
        blocker = self.holding("blocker", 4)
        waiting = self.start("-c", 1, "-l", "waiting", "--", *self.held("waiting", "sleep:0"), name="waiting")
        self.wait_for(lambda: self.queued() == 1, "the run to queue")
        waiting.send_signal(signal.SIGINT)
        self.assertEqual(self.finish(waiting).code, 130)
        self.assertEqual(os.listdir(self.state / "waiters"), [])
        self.end(blocker)
        self.assertEqual(self.gate("-c", 1, "--", PY, "-c", "pass").code, 0)
        self.assertFalse(self.started("waiting"))
        self.assertEqual([(e["label"], e["reason"]) for e in self.events("not-admitted")], [("waiting", "interrupted")])
        self.assertEqual(self.events("abandoned"), [])

    def test_wait_limits_give_up_without_running(self):
        blocker = self.holding("blocker", 4)
        result = self.gate("--no-wait", "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("nothing ran", result.err)
        result = self.gate("--wait", 1, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertTrue(0.9 <= result.elapsed < 6, result.elapsed)
        # Behind another waiter, the limit still holds.
        head = self.start("-c", 1, "--", *self.held("head", "sleep:0"), name="head")
        self.wait_for(lambda: self.queued() == 1, "the head to queue")
        result = self.gate("--wait", 1, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("earlier", result.err)
        self.assertFalse(self.started("never"))
        self.assertEqual(len(self.events("not-admitted")), 3)
        self.release("blocker")
        self.assertEqual(self.finish(head).code, 0)
        self.assertEqual(self.finish(blocker).code, 0)
        self.assertEqual(os.listdir(self.state / "waiters"), [])


    def test_wait_limits_cover_queue_registration(self):
        self.state.mkdir(mode=0o700)
        fd = os.open(self.state / "queue.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            result = self.gate("--no-wait", "-c", 1, "--", *self.held("never", "sleep:0"))
            self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
            self.assertLess(result.elapsed, 4)
            result = self.gate("--wait", 2, "-c", 1, "--", *self.held("never", "sleep:0"))
            self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
            self.assertTrue(1.9 <= result.elapsed < 6, result.elapsed)
            self.assertIn("registration", result.err)
        finally:
            os.close(fd)
        self.assertFalse(self.started("never"))
        self.assertEqual(self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass").code, 0)


    def test_the_wait_limit_leaves_out_collecting_dead_gates_runs(self):
        self.ensure_state()
        # Unique: the sweep finds survivors host-wide by tag, so a fixed one would claim another suite's.
        run_id = f"{random.getrandbits(64):016x}"
        stubborn = self.sleeper("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                                env=dict(self.env, AGENT_GATE_RUN=run_id), start_new_session=True)
        self.wait_for(lambda: gate_macos.tagged_pids(run_id), "the survivor's tag")
        os.close(agent_gate.publish_locked(self.state / "jobs", f"{run_id}.lock", {
            "id": run_id, "k": 4, "pid": stubborn.pid, "start": 1, "admitted": time.time(), "label": "dead"}))
        # Stopping the survivor takes the TERM grace; that is housekeeping, not waiting for capacity.
        result = self.gate("--no-wait", "-c", 4, "--", *self.held("after", "sleep:0"))
        self.assertEqual(result.code, 0, result.err)
        self.assertGreaterEqual(result.elapsed, agent_gate.GRACE_SECONDS)
        self.assertIsNotNone(stubborn.poll())

    def test_waiters_behind_the_head_collect_dead_gates_runs(self):
        lock = self.tmp / "ahead.lock"
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX)
        calls: List[float] = []
        original = agent_gate.SWEEP_SECONDS, agent_gate.WAITER_POLL_SECONDS
        agent_gate.SWEEP_SECONDS, agent_gate.WAITER_POLL_SECONDS = 0.2, 0.05
        try:
            with self.assertRaises(agent_gate.WaitExpired):
                agent_gate.block_shared(str(lock), time.monotonic() + 1.5, lambda: calls.append(time.monotonic()))
        finally:
            agent_gate.SWEEP_SECONDS, agent_gate.WAITER_POLL_SECONDS = original
        self.assertGreaterEqual(len(calls), 3)


class OffSwitchTest(GateCase):
    def test_off_runs_the_command_as_if_there_were_no_gate_and_on_resumes_the_queue(self):
        self.assertEqual(self.gate("off").code, 0)
        self.assertTrue(self.status()["off"])
        names = (agent_gate.CPUS_ENV, agent_gate.RUN_ENV, *agent_gate.RUNNER_ENV)
        probe = f"import os, time; time.sleep(2); print([name for name in {names!r} if name in os.environ])"
        result = self.gate("-c", 4, "--timeout", 1, "--", PY, "-c", probe)
        self.assertEqual((result.code, result.out.strip()), (0, "[]"), "no worker limits, run tag or fuse")
        blocker = self.holding("blocker", 4)
        beside = self.holding("beside", 4)
        self.assertEqual(self.queued(), 0, "nothing queues while the gate is off")
        self.assertEqual(self.jobs(), [], "a run started while off leaves no record")
        inside = subprocess.run([PY, str(GATE), "inside", "-c", "4"], env=self.env)
        self.assertEqual(inside.returncode, 0, "while off, every caller may run at once")
        self.assertEqual(self.gate("on").code, 0)
        self.assertFalse(self.status()["off"])
        after = self.start("-c", 1, "--", *self.held("after", "sleep:0"), name="after")
        self.assertEqual(self.finish(after).code, 0)
        self.end(blocker, beside)
        self.assertEqual(len(self.events("admitted")), 1, "only the run started after `on` was admitted")
        self.assertEqual(os.listdir(self.state / "jobs"), [])

    def test_off_starts_a_request_already_queued(self):
        blocker = self.start("-c", 4, "-l", "blocker", "--", *self.held("blocker"), name="blocker")
        self.command_pid("blocker")
        queued = self.start("-c", 1, "-l", "queued", "--", *self.held("queued"), name="queued")
        self.wait_for(lambda: "4 of 4 tokens in use, 1 needed" in self.err("queued"), "the request to wait")
        switched = time.monotonic()
        self.assertEqual(self.gate("off").code, 0)
        self.command_pid("queued")
        self.assertLess(time.monotonic() - switched, 2.0, "the queued request starts once the gate is off")
        self.assertEqual(self.queued(), 0, "it left the queue")
        self.assertEqual([job.get("label") for job in self.jobs()], ["blocker"], "it leaves no record")
        self.assertEqual(self.status()["used"], 4, "the blocker still holds its tokens")
        self.end(blocker, queued)
        self.assertEqual([event["label"] for event in self.events("released")], ["blocker"])


class MemoryTest(GateCase):
    def test_critical_pressure_holds_the_queue_until_it_clears(self):
        self.set_memory("4 80")
        proc = self.start("-c", 1, "--", *self.held("x", "sleep:0"), name="x")
        self.wait_for(lambda: "memory pressure is critical" in self.err("x"), "the critical refusal")
        time.sleep(0.5)
        self.assertFalse(self.started("x"))
        self.set_memory("1 80")
        self.assertEqual(self.finish(proc).code, 0)

    def test_the_log_keeps_why_the_head_waited(self):
        self.set_memory("4 80")
        proc = self.start("-c", 1, "--", *self.held("x", "sleep:0"), name="x")
        self.wait_for(lambda: "memory pressure is critical" in self.err("x"), "the critical refusal")
        time.sleep(0.5)
        self.set_memory("1 80")
        self.assertEqual(self.finish(proc).code, 0)
        [admitted] = self.events("admitted")
        self.assertIn("memory pressure is critical", admitted["held_back"])
        self.assertGreaterEqual(admitted["head_s"], 0.5)

    def test_a_run_admitted_at_once_names_no_hold_back(self):
        self.assertEqual(self.finish(self.start("-c", 1, "--", "true", name="x")).code, 0)
        [admitted] = self.events("admitted")
        self.assertNotIn("held_back", admitted)
        self.assertLess(admitted["head_s"], 5)

    def test_an_unreadable_memory_signal_allows_one_run_at_a_time(self):
        self.set_memory("unavailable")
        blocker = self.holding("blocker", 1)
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("memory signal unavailable", result.err)
        self.end(blocker)

    def test_an_unreadable_pressure_level_still_admits_runs_side_by_side(self):
        self.set_memory("- 80")
        blocker = self.holding("blocker", 1)
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        self.end(blocker)

    def test_low_free_memory_holds_a_second_run_but_not_a_one_token_one(self):
        self.set_memory("2 10")
        blocker = self.holding("blocker", 1)
        result = self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("10% memory free", result.err)
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        self.set_memory("2 30")
        self.assertEqual(self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass").code, 0)
        self.end(blocker)

    def test_swapping_holds_a_second_run_until_it_stops(self):
        self.set_memory("2 80 50")
        blocker = self.holding("blocker", 1)
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("swaps out 50.0 MiB/s", result.err)
        self.set_memory("2 80 0")
        self.assertEqual(self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass").code, 0)
        self.end(blocker)

    def test_admissions_are_spaced_by_the_ramp_only_while_free_memory_is_thin(self):
        # 25% minimum and a 10-point margin: the ramp holds below 35% free.
        env = dict(self.env, AGENT_GATE_RAMP_SECONDS="3")
        first = self.start("-c", 1, "--", *self.held("first"), env=env, name="first")
        self.command_pid("first")
        result = self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass", env=env)
        self.assertEqual(result.code, 0, f"80% free: no ramp. {result.err}")
        self.set_memory("1 30")
        result = self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass", env=env)
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("still starting", result.err)
        result = self.gate("--wait", 10, "-c", 2, "--", PY, "-c", "pass", env=env)
        self.assertEqual(result.code, 0, result.err)
        first_at, plenty, thin = (event["admitted"] for event in self.events("admitted"))
        self.assertLess(plenty - first_at, 3, "admitted without the ramp")
        self.assertGreaterEqual(thin - plenty, 2.9, "spaced by the ramp")
        self.assertEqual([event["ramp"] for event in self.events("admitted")], [False, False, True])
        self.end(first)


#: A gate that notes each reading of the host in $SLOW_GATE_MARKS, then pauses
#: $SLOW_GATE_SECONDS before acting on it.
SLOW_GATE = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, sys.argv.pop(1))
    import agent_gate

    read = agent_gate.read_snapshot

    def slow(*args, **kwargs):
        snapshot = read(*args, **kwargs)
        with open(os.environ["SLOW_GATE_MARKS"], "a") as marks:
            marks.write(f"{snapshot.used:g}\\n")
        time.sleep(float(os.environ["SLOW_GATE_SECONDS"]))
        return snapshot

    agent_gate.read_snapshot = slow
    raise SystemExit(agent_gate.main())
    """
)


#: A gate that, once admitted, marks $PAUSE_GATE_MARK and pauses $PAUSE_GATE_SECONDS before its command starts.
PAUSE_GATE = textwrap.dedent(
    """
    import os, sys, time
    sys.path.insert(0, sys.argv.pop(1))
    import agent_gate

    environment = agent_gate.runner_environment

    def paused(*args, **kwargs):
        open(os.environ["PAUSE_GATE_MARK"], "w").close()
        time.sleep(float(os.environ["PAUSE_GATE_SECONDS"]))
        return environment(*args, **kwargs)

    agent_gate.runner_environment = paused
    raise SystemExit(agent_gate.main())
    """
)


class BackfillTest(GateCase):
    """Later requests start beside earlier ones that wait for tokens; 1-token runs have headroom."""

    tokens = 10

    def setUp(self) -> None:
        super().setUp()
        self.env.update(AGENT_GATE_MAX_CLAIM="5", AGENT_GATE_BACKFILL_SECONDS="60")

    def test_a_small_run_starts_while_the_head_waits_for_tokens(self):
        big, mid = self.holding("big", 5), self.holding("mid", 3)
        head = self.start("-c", 5, "--", *self.held("head", "sleep:0"), name="head")
        self.wait_for(lambda: "8 of 10 tokens in use, 5 needed" in self.err("head"), "the head to wait for tokens")
        for k in (1, 2):
            result = self.gate("--no-wait", "-c", k, "-l", f"k{k}", "--", PY, "-c", "pass")
            self.assertEqual(result.code, 0, f"k={k} fits beside the head's wait: {result.err}")
        result = self.gate("--no-wait", "-c", 3, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("8 of 10 tokens in use, 3 needed", result.err)
        self.assertFalse(self.started("head"))
        passed = {event["label"]: event for event in self.events("admitted") if event["label"] in ("k1", "k2")}
        self.assertEqual({label: (event["backfilled"], event["headroom"]) for label, event in passed.items()},
                         {"k1": (True, False), "k2": (True, False)})
        self.end(big)
        self.assertEqual(self.finish(head).code, 0)
        self.assertFalse(self.events("admitted")[-1]["backfilled"], "the head took its own turn")
        self.end(mid)

    def test_an_aged_request_is_not_passed(self):
        self.env["AGENT_GATE_BACKFILL_SECONDS"] = "1"
        big, mid = self.holding("big", 5), self.holding("mid", 3)
        head = self.start("-c", 5, "--", *self.held("head", "sleep:0"), name="head")
        self.wait_for(lambda: self.status()["reserved"] == 5, "the head to age")
        result = self.gate("--no-wait", "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("waited over 1s for its tokens; nothing passes it except a 1-token run into the headroom "
                      "while runs outside it hold all 10 tokens", result.err)
        text = self.status_text()
        self.assertIn("5 tokens reserved for 1 request(s) queued over 1s; only a 1-token run into the headroom "
                      "passes them, while runs outside it hold all 10 tokens (8 now)", text)
        self.assertFalse(self.started("never"))
        self.end(big)
        self.assertEqual(self.finish(head).code, 0)
        self.end(mid)

    def test_a_small_run_passes_an_aged_request_only_while_runs_outside_the_headroom_fill_the_pool(self):
        self.env.update(AGENT_GATE_BACKFILL_SECONDS="1", AGENT_GATE_SMALL_HEADROOM="2")
        five, four, one = self.holding("five", 5), self.holding("four", 4), self.holding("one", 1)
        head = self.start("-c", 5, "--", *self.held("head", "sleep:0"), name="head")
        self.wait_for(lambda: self.status()["reserved"] == 5, "the head to age")
        first = self.holding("first", 1)
        report = self.status()
        self.assertEqual((report["used"], report["outside_headroom"]), (11, 10), "the first took the headroom")
        self.end(first)
        # A stream: the headroom run ended, but runs outside the headroom still hold all 10 tokens.
        second = self.holding("second", 1)
        self.assertEqual(self.status()["used"], 11)
        self.end(one)
        report = self.status()
        self.assertEqual((report["used"], report["outside_headroom"]), (10, 9))
        result = self.gate("--no-wait", "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, f"a run outside the headroom ended: {result.err}")
        self.assertIn("waited over 1s for its tokens; nothing passes it", result.err)
        self.assertFalse(self.started("never"))
        self.assertFalse(self.started("head"))
        self.assertEqual([(event["backfilled"], event["headroom"]) for event in self.events("admitted")[3:]],
                         [(True, True), (True, True)])
        self.end(second, four)
        self.assertEqual(self.finish(head).code, 0)
        self.assertFalse(self.events("admitted")[-1]["backfilled"], "the head took its own turn")
        self.end(five)

    def test_a_small_run_takes_the_headroom_when_the_pool_is_full(self):
        self.env["AGENT_GATE_SMALL_HEADROOM"] = "2"
        first, second = self.holding("first", 5), self.holding("second", 5)
        small = self.holding("small", 1)
        result = self.gate("--no-wait", "-c", 2, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("11 of 10 tokens in use, 2 needed", result.err)
        other = self.holding("other", 1)
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("12 of 10 tokens in use (+2 for 1-token runs), 1 needed", result.err)
        self.assertEqual([event["headroom"] for event in self.events("admitted")], [False, False, True, True])
        report = self.status()
        self.assertEqual((report["used"], report["headroom"]), (12, 2))
        text = self.status_text()
        self.assertIn("12 of 10 tokens in use (+2 for 1-token runs; runs admitted into it hold 2)", text)
        self.end(first, second, small, other)

    def test_status_counts_the_live_runs_in_the_headroom(self):
        self.env["AGENT_GATE_SMALL_HEADROOM"] = "2"
        first, second = self.holding("first", 5), self.holding("second", 5)
        small = self.holding("small", 1)
        self.end(second)
        # The small run keeps its headroom place though the pool is no longer full.
        report = self.status()
        self.assertEqual((report["used"], report["headroom_used"], report["outside_headroom"]), (6, 1, 5))
        text = self.status_text()
        self.assertIn("6 of 10 tokens in use (+2 for 1-token runs; runs admitted into it hold 1)", text)
        self.end(first, small)

    def test_concurrent_admissions_stay_within_the_limits(self):
        # 6 tokens and 1 of headroom.
        self.env.update(AGENT_GATE_TOKENS="6", AGENT_GATE_MAX_CLAIM="3", AGENT_GATE_SMALL_HEADROOM="1")
        rng = random.Random(11)
        jobs = [(f"j{index}", rng.choice((1, 1, 1, 2, 3)), rng.uniform(0.5, 1.2)) for index in range(16)]
        procs = [self.start("-c", k, "-l", name, "--", *self.held(name, f"sleep:{seconds:.2f}"), name=name)
                 for name, k, seconds in jobs]
        for proc in procs:
            self.assertEqual(self.finish(proc, 120).code, 0, self.err(proc.name))  # type: ignore[attr-defined]
        tokens = {name: k for name, k, _ in jobs}
        ends = self.times("end")
        # A command marks its start after admission and its end before release, so the marks
        # show each run for less than it held its tokens.
        timeline = [(at, 1, name) for name, at in self.times("start").items()]
        timeline += [(at, 0, name) for name, at in ends.items()]
        running: Dict[str, int] = {}
        most = 0
        for _, starting, name in sorted(timeline):
            if not starting:
                del running[name]
                continue
            running[name] = tokens[name]
            most = max(most, len(running))
            self.assertLessEqual(sum(running.values()), 7, f"{name} started beside {running}: beyond the headroom")
        # Admissions are stamped under the admit lock, so in order: each saw every earlier run
        # that had not yet marked its end.
        admitted = {str(event["label"]): float(event["admitted"]) for event in self.events("admitted")}
        for name, at in admitted.items():
            held = {other: tokens[other] for other, then in admitted.items() if then < at and ends[other] > at}
            limit = 7 if tokens[name] == 1 else 6
            self.assertLessEqual(sum(held.values()) + tokens[name], limit,
                                 f"{name} (k={tokens[name]}) admitted while {held} held tokens")
        self.assertGreater(most, 2, "runs that fit together should overlap")

    def test_a_decision_and_its_run_are_published_together(self):
        # 6 tokens: a 3-token head waits while 4 are in use, and a 2-token request decides to pass it.
        self.env.update(AGENT_GATE_TOKENS="6", AGENT_GATE_MAX_CLAIM="3")
        program = self.tmp / "slow-gate.py"
        program.write_text(SLOW_GATE)
        readings = self.tmp / "readings"
        big, small = self.holding("big", 3), self.holding("small", 1)
        head = self.start("-c", 3, "--", *self.held("head"), name="head")
        self.wait_for(lambda: "4 of 6 tokens in use, 3 needed" in self.err("head"), "the head to wait for tokens")
        passer = self.start(str(ROOT / "lib"), "-c", 2, "--", *self.held("passer"), name="passer", program=program,
                            env=dict(self.env, SLOW_GATE_MARKS=str(readings), SLOW_GATE_SECONDS="2"))
        # The passer reads the host once to decide and once more to publish on; the head fits once
        # `small` ends while the passer still holds the second reading.
        self.wait_for(lambda: readings.exists() and len(readings.read_text().split()) >= 2, "the passer's decision")
        self.end(small)
        self.command_pid("passer")
        time.sleep(1)
        self.assertFalse(self.started("head"), "the head started on a reading that missed the passer's run")
        self.assertEqual(self.status()["used"], 5)
        self.end(passer)
        self.wait_for(lambda: self.started("head"), "the head to start once the passer ends")
        self.end(head, big)


class NestingTest(GateCase):
    def test_a_nested_gate_runs_at_once_within_the_parent_budget(self):
        other = self.holding("other", 2)
        probe = [PY, "-c", "import os; print(os.environ['AGENT_GATE_CPUS'], os.environ['VITEST_MAX_WORKERS'])"]
        script = self.gate_script(
            f"""
            for k in ("4", "1"):
                result = subprocess.run(gate + ["--no-wait", "-c", k, "--"] + {probe!r}, capture_output=True, text=True)
                print(k, result.returncode, result.stdout.strip(), flush=True)
            print("inside", subprocess.run(gate + ["inside"]).returncode, flush=True)
            """
        )
        result = self.gate("-c", 2, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.splitlines(), ["4 0 2 2", "1 0 1 1", "inside 0"])
        parents = {event["parent"] for event in self.events("nested")}
        self.assertEqual(sorted((event["asked"], event["k"]) for event in self.events("nested")), [(1, 1), (4, 2)])
        self.assertEqual(len(parents), 1)
        self.assertEqual(len(self.events("admitted")), 2, "nested runs are never admitted separately")
        self.end(other)

    def test_a_nested_budget_only_shrinks(self):
        probe = [PY, "-c", "import os; print(os.environ['AGENT_GATE_CPUS'])"]
        inner = self.gate_script(
            f"""
            print(subprocess.run(gate + ["--no-wait", "-c", "4", "--"] + {probe!r}, capture_output=True,
                                 text=True).stdout.strip(), flush=True)
            print(subprocess.run(gate + ["inside", "-c", "2"]).returncode, flush=True)
            print(subprocess.run(gate + ["inside", "-c", "1"]).returncode, flush=True)
            """
        )
        outer = self.gate_script(
            f"""
            subprocess.run(gate + ["--no-wait", "-c", "1", "--", sys.executable, "-c", {inner!r}])
            """
        )
        result = self.gate("-c", 4, "--", PY, "-c", outer)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.splitlines(), ["1", str(agent_gate.CONFIG_EXIT), "0"])

    def test_inside_judges_the_needed_claim_after_the_same_cap_a_claim_gets(self):
        # A consumer that asked -c 4 was admitted capped to 2; its own `inside -c 4` must not abort it.
        env = dict(self.env, AGENT_GATE_MAX_CLAIM="2")
        script = self.gate_script(
            """
            for k in ("4", "2", "1"):
                print(k, subprocess.run(gate + ["inside", "-c", k]).returncode, flush=True)
            """
        )
        result = self.gate("-c", 4, "--", PY, "-c", script, env=env)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(
            result.out.splitlines(), ["4 0", "2 0", "1 0"],
            "`inside -c N` must judge N after the same cap a claim gets (AGENT_GATE_MAX_CLAIM) before comparing "
            f"it with the run's budget: {result.err}")
        # The cap does not widen a narrower budget: a parent admitted with -c 1 still refuses -c 2.
        narrow = self.gate("-c", 1, "--", PY, "-c", script, env=env)
        self.assertEqual(narrow.out.splitlines(), ["4 78", "2 78", "1 0"], narrow.err)

    def test_inside_still_refuses_a_need_above_the_budget_when_the_cap_is_higher(self):
        env = dict(self.env, AGENT_GATE_MAX_CLAIM="3")
        script = self.gate_script("""print(subprocess.run(gate + ["inside", "-c", "4"]).returncode)""")
        result = self.gate("-c", 2, "--", PY, "-c", script, env=env)
        self.assertEqual(result.out.split(), [str(agent_gate.CONFIG_EXIT)], result.err)

    def test_a_nested_command_carries_its_parents_run_tag(self):
        probe = [PY, "-c", "import os; print(os.environ.get('AGENT_GATE_RUN'))"]
        script = self.gate_script(
            f"""
            gate = gate + ["--no-wait", "-c", "1", "--"] + {probe!r}
            print(os.environ["AGENT_GATE_RUN"], flush=True)
            missing = {{k: v for k, v in os.environ.items() if k != "AGENT_GATE_RUN"}}
            stale = dict(os.environ, AGENT_GATE_RUN="0123456789abcdef")
            for env in (missing, stale):
                result = subprocess.run(gate, env=env, capture_output=True, text=True, close_fds=True)
                print(result.returncode, result.stdout.strip(), flush=True)
            """
        )
        result = self.gate("-c", 2, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        parent, *nested = result.out.splitlines()
        self.assertEqual(nested, [f"0 {parent}", f"0 {parent}"])

    def test_a_leaked_run_tag_without_a_live_ancestor_does_not_bypass(self):
        blocker = self.holding("blocker", 4)
        run_id = str(self.jobs()[0]["id"])
        leaked = dict(
            self.env,
            AGENT_GATE_RUN=run_id,
            AGENT_GATE_CPUS="4",
        )
        result = self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass", env=leaked)
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        inside = subprocess.run([PY, str(GATE), "inside"], env=leaked)
        self.assertEqual(inside.returncode, 1)
        self.end(blocker)


class ProcessTest(GateCase):
    def test_the_command_tree_gives_way_to_the_owners_apps(self):
        probe = ("import os, subprocess, sys, time; time.sleep(0.2); "
                 "subprocess.run([sys.executable, '-c', 'import os; print(os.getpriority(os.PRIO_PROCESS, 0))'])")
        result = self.gate("--", PY, "-c", probe)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.split(), [str(gate_macos.RUN_NICENESS)])

    def test_the_fuse_stops_a_run_and_exits_124(self):
        result = self.gate("--timeout", 1, "--", PY, "-c", "import time; time.sleep(30)")
        self.assertEqual(result.code, agent_gate.TIMEOUT_EXIT)
        self.assertLess(result.elapsed, 5)
        pidfile = self.tmp / "stubborn.pid"
        stubborn = (
            "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        )
        result = self.gate("--timeout", 1, "--", PY, "-c", stubborn)
        self.assertEqual(result.code, agent_gate.TIMEOUT_EXIT)
        self.assertTrue(5 <= result.elapsed < 12, result.elapsed)
        self.assertTrue(self.gone(int(pidfile.read_text())))
        self.assertTrue(all(event["timed_out"] for event in self.events("released")))

    def test_signals_to_the_gate_reach_the_command(self):
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                name = signum.name.lower()
                proc = self.start("-c", 1, "--", *self.held(name), name=name)
                pid = self.command_pid(name)
                proc.send_signal(signum)
                result = self.finish(proc)
                self.assertEqual(result.code, 128 + signum)
                self.assertTrue(self.gone(pid))
                self.assertEqual(os.listdir(self.state / "jobs"), [])
        self.assertTrue(all(event["cancelled"] for event in self.events("released")))

    def test_signals_the_caller_ignores_stay_ignored_for_the_command(self):
        # As under nohup, or in the background of a shell without job control.
        program = self.tmp / "ignoring.py"
        program.write_text(textwrap.dedent(
            f"""
            import os, signal, sys
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            os.execv(sys.executable, [sys.executable, {str(GATE)!r}, *sys.argv[1:]])
            """
        ))
        probe = ("import signal; print(*(getattr(signal.getsignal(s), 'name', 'handled') "
                 "for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)))")
        result = self.gate("-c", 1, "--", PY, "-c", probe, program=program)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.split(), ["SIG_IGN", "SIG_DFL", "SIG_IGN"])

    def test_stop_signals_during_the_sweep_still_release_the_run(self):
        pidfile = self.tmp / "stubborn.pid"
        stubborn = (
            "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
        )
        # The command exits at once with 3; its leftover ignores TERM, so the sweep takes the grace period.
        script = textwrap.dedent(
            f"""
            import subprocess, sys, time
            from pathlib import Path
            subprocess.Popen([sys.executable, "-c", {stubborn!r}], start_new_session=True)
            while not (Path({str(pidfile)!r}).exists() and Path({str(pidfile)!r}).read_text()):
                time.sleep(0.02)
            open({str(self.tmp / "command.pid")!r}, "w").write(str(__import__("os").getpid()))
            raise SystemExit(3)
            """
        )
        proc = self.start("-c", 1, "--", PY, "-c", script, name="sweeping")
        self.wait_for(lambda: (self.tmp / "command.pid").exists() and (self.tmp / "command.pid").read_text(),
                      "the command to end")
        self.leftovers.append(int(pidfile.read_text()))
        self.wait_for(lambda: self.gone(int((self.tmp / "command.pid").read_text())), "the command to be reaped")
        time.sleep(1)
        self.assertIsNone(proc.poll(), "the gate is still sweeping")
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            proc.send_signal(signum)
            time.sleep(0.2)
        result = self.finish(proc)
        self.assertEqual(result.code, 3, result.err)
        self.assertNotIn("Traceback", result.err)
        [released] = self.events("released")
        self.assertEqual((released["exit"], released["swept"], released["survivors"], released["cancelled"]),
                         (3, 1, 0, False))
        self.assertTrue(self.gone(int(pidfile.read_text())))
        self.assertEqual(os.listdir(self.state / "jobs"), [])

    def test_a_stop_signal_before_the_command_starts_means_it_never_does(self):
        program = self.tmp / "pause-gate.py"
        program.write_text(PAUSE_GATE)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                mark = self.tmp / f"paused-{signum.name}"
                ran = self.tmp / f"ran-{signum.name}"
                env = dict(self.env, PAUSE_GATE_MARK=str(mark), PAUSE_GATE_SECONDS="1")
                proc = self.start(str(ROOT / "lib"), "-c", 1, "--", "/usr/bin/touch", ran, name=signum.name,
                                  program=program, env=env)
                self.wait_for(mark.exists, "the gate to be admitted")
                proc.send_signal(signum)
                result = self.finish(proc)
                self.assertEqual(result.code, 128 + signum, result.err)
                self.assertNotIn("Traceback", result.err)
                self.assertFalse(ran.exists(), "the command started")
                released = self.events("released")[-1]
                self.assertEqual((released["exit"], released["cancelled"], released["signal"], released["cpus"]),
                                 (128 + signum, True, None, None))
                self.assertEqual(os.listdir(self.state / "jobs"), [])
        self.assertEqual(len(self.events("admitted")), len(self.events("released")))
        self.assertEqual(self.events("not-admitted"), [])

    def test_killing_the_callers_group_stops_the_command_and_frees_its_tokens(self):
        proc = self.start("-c", 4, "--", *self.held("x"), name="x")
        pid = self.command_pid("x")
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        self.wait_for(lambda: self.gone(pid), "the command to die with its group")
        self.assertEqual(self.gate("--no-wait", "-c", 4, "--", PY, "-c", "pass").code, 0)
        self.assertEqual(os.listdir(self.state / "jobs"), [])

    def test_a_killed_gate_keeps_its_tokens_until_the_next_head_stops_the_command(self):
        pidfile = self.tmp / "sleep.pid"
        commands = {
            # A runner's environment is visible, so its run tag finds it.
            "tagged": self.held("tagged"),
            # A platform binary hides its environment; only its job lock gives it away.
            "lock holder": ["/bin/sh", "-c", f'echo $$ > "{pidfile}"; exec /bin/sleep 3600'],
        }
        for kind, command in commands.items():
            with self.subTest(kind):
                proc = self.start("-c", 4, "--", *command, name=kind.replace(" ", "-"))
                if kind == "tagged":
                    pid = self.command_pid("tagged")
                else:
                    self.wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), "sleep to start")
                    pid = int(pidfile.read_text())
                    self.leftovers.append(pid)
                    self.wait_for(lambda: "sleep" in command_of(pid), "the exec")
                proc.kill()
                proc.wait()
                report = self.status()
                self.assertEqual(report["used"], 4)
                self.assertFalse(report["running"][0]["gate_alive"])
                self.assertFalse(self.gone(pid), "the command outlives its gate")
                result = self.gate("--wait", 30, "-c", 4, "--", PY, "-c", "pass")
                self.assertEqual(result.code, 0, result.err)
                self.assertTrue(self.gone(pid))
                self.assertEqual(os.listdir(self.state / "jobs"), [])
        self.assertEqual(len(self.events("orphan-stopped")), 2)

    def test_tagged_leftovers_are_swept_when_the_command_exits(self):
        pidfile = self.tmp / "leftover.pid"
        script = textwrap.dedent(
            f"""
            import subprocess, sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            open({str(pidfile)!r}, "w").write(str(child.pid))
            """
        )
        result = self.gate("-c", 1, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        pid = int(pidfile.read_text())
        self.leftovers.append(pid)
        released = self.events("released")[-1]
        self.assertTrue(self.gone(pid), (released, gate_macos.process_info(pid), command_of(pid)))
        self.assertEqual(released["swept"], 1)
        self.assertFalse(released["left_held"])

    def test_a_process_in_exec_is_read_again(self):
        run_id = f"{random.getrandbits(64):016x}"
        child = self.sleeper(env=dict(self.env, AGENT_GATE_RUN=run_id), start_new_session=True)
        self.wait_for(lambda: gate_macos.tagged_pids(run_id), "the child's tag")
        read, unreadable = gate_macos._procargs, [3]

        def in_exec(pid: int, buffer: object) -> object:
            # Exec replaces the arguments: for a moment the kernel has none to give.
            if pid == child.pid and unreadable[0]:
                unreadable[0] -= 1
                return None
            return read(pid, buffer)

        with unittest.mock.patch.object(gate_macos, "_procargs", in_exec):
            self.assertEqual(list(gate_macos.tagged_pids(run_id) or {}), [child.pid])
        self.assertEqual(unreadable, [0])

    def test_a_leftover_that_replaces_itself_while_stopping_is_swept_too(self):
        pidfile = self.tmp / "replacement.pid"
        replacing = textwrap.dedent(
            f"""
            import os, signal, subprocess, sys, time
            def replace(*_):
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                         start_new_session=True)
                open({str(pidfile)!r}, "w").write(str(child.pid))
                os._exit(0)
            signal.signal(signal.SIGTERM, replace)
            time.sleep(60)
            """
        )
        script = textwrap.dedent(
            f"""
            import subprocess, sys
            subprocess.Popen([sys.executable, "-c", {replacing!r}], start_new_session=True)
            import time; time.sleep(1)
            """
        )
        result = self.gate("-c", 1, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        pid = int(pidfile.read_text())
        self.leftovers.append(pid)
        self.assertTrue(self.gone(pid))
        released = self.events("released")[-1]
        self.assertEqual((released["swept"], released["survivors"], released["left_held"]), (2, 0, False))
        self.assertEqual(os.listdir(self.state / "jobs"), [])

    def test_a_run_whose_leftovers_survive_keeps_its_record(self):
        self.ensure_state()
        settings = self.settings()
        request = agent_gate.Request(2, "kept", ["true"], None, 0)
        record = {"id": request.id, "k": 2, "pid": os.getpid(), "start": gate_macos.own_start(),
                  "admitted": time.time()}
        job = agent_gate.publish_locked(settings.jobs, f"{request.id}.lock", record)
        self.assertTrue(agent_gate._release(settings, request.id, job, keep=True))
        self.assertTrue((settings.jobs / f"{request.id}.lock").exists())

    def test_an_untagged_leftover_holding_the_job_lock_is_stopped_by_the_next_head(self):
        pidfile = self.tmp / "sleep.pid"
        script = textwrap.dedent(
            f"""
            import os, subprocess
            # Drop the run tag: macOS hides a platform binary's environment only
            # while SIP is on, and CI runners may have it off.
            env = {{k: v for k, v in os.environ.items() if k != {agent_gate.RUN_ENV!r}}}
            child = subprocess.Popen(["/bin/sleep", "3600"], close_fds=False, start_new_session=True, env=env)
            open({str(pidfile)!r}, "w").write(str(child.pid))
            """
        )
        result = self.gate("-c", 4, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        pid = int(pidfile.read_text())
        self.leftovers.append(pid)
        self.assertFalse(self.gone(pid))
        self.assertTrue(self.events("released")[-1]["left_held"])
        result = self.gate("--wait", 30, "-c", 4, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        self.assertTrue(self.gone(pid))

    def test_survivors_the_head_may_not_signal_keep_their_tokens(self):
        pidfile = self.tmp / "survivor.pid"
        script = textwrap.dedent(
            f"""
            import subprocess, sys
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
            open({str(pidfile)!r}, "w").write(str(child.pid))
            """
        )
        # The gate dies before its own sweep; its tagged child closed the job lock.
        proc = self.start("-c", 4, "--", PY, "-c", script + "\nimport time; time.sleep(60)", name="dying")
        self.wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), "the child to start")
        pid = int(pidfile.read_text())
        self.leftovers.append(pid)
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        self.wait_for(lambda: self.jobs() and not self.jobs()[0]["held"], "the run's own processes to end")
        settings = self.settings()
        real_kill = os.kill

        def refused(target: int, signum: int) -> None:
            if target == pid:
                raise PermissionError(1, "Operation not permitted")
            real_kill(target, signum)

        os.kill = refused
        try:
            jobs = agent_gate.scan_jobs(settings, sweep=True)
        finally:
            os.kill = real_kill
        self.assertEqual([job["k"] for job in jobs], [4], "a survivor keeps the run's tokens")
        self.assertFalse(self.gone(pid))
        self.assertEqual(self.events("orphan-survives")[-1]["remaining"], 1)
        # A head outside the sandbox, its pause over.
        (settings.jobs / f"{jobs[0]['id']}.retry").unlink()
        self.assertEqual(agent_gate.scan_jobs(settings, sweep=True), [])
        self.assertTrue(self.gone(pid))
        self.assertEqual(os.listdir(self.state / "jobs"), [])

    def test_a_dead_gates_record_is_retried_soon_after_its_holder_lets_go(self):
        self.ensure_state()
        settings = self.settings()
        ended = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        record = {"id": "00000000deadbeef", "k": 2, "pid": int(ended.stdout), "start": 1,
                  "admitted": time.time(), "label": "dead gate"}
        # This process stands in for a holder still exiting: lsof never reports the caller itself.
        fd = agent_gate.publish_locked(self.state / "jobs", "00000000deadbeef.lock", record)
        try:
            self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [2])
        finally:
            os.close(fd)
        self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [2], "attempts are spaced")
        time.sleep(1.1)
        self.assertEqual(agent_gate.scan_jobs(settings, sweep=True), [])
        self.assertEqual([event["event"] for event in self.events("orphan-survives") + self.events("orphan-stopped")],
                         ["orphan-survives", "orphan-stopped"])

    def test_a_failed_attempt_spaces_the_next_one_for_every_gate(self):
        self.ensure_state()
        settings = self.settings()
        ended = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        record = {"id": "00000000feedface", "k": 2, "pid": int(ended.stdout), "start": 1,
                  "admitted": time.time(), "label": "dead gate"}
        # This process holds the job lock, so every attempt fails until it lets go.
        fd = agent_gate.publish_locked(settings.jobs, "00000000feedface.lock", record)
        retry = settings.jobs / "00000000feedface.retry"
        # After five failures the next pauses 32 s, however slowly the other gate starts.
        retry.write_text(json.dumps({"failures": 5, "next": 0}))
        try:
            self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [2])
            self.assertAlmostEqual(json.loads(retry.read_text())["next"] - time.time(), 32, delta=2)
            other_gate = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(ROOT / "lib")!r})
                import agent_gate
                settings = agent_gate.Settings({{"AGENT_GATE_DIR": {str(self.state)!r}}})
                print(len(agent_gate.scan_jobs(settings, sweep=True)))
                """
            )
            result = subprocess.run([PY, "-c", other_gate], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.stdout.split(), ["1"], result.stderr)
            self.assertEqual(len(self.events("orphan-survives")), 1, "another gate retried at once")
            # Older gates' scans pass over the retry state: their job and staging patterns are these.
            self.assertEqual(sorted(os.listdir(settings.jobs)), ["00000000feedface.lock", retry.name])
            self.assertFalse(agent_gate.JOB_RE.match(retry.name) or agent_gate.STAGING_RE.match(retry.name))
        finally:
            os.close(fd)
        retry.write_text(json.dumps({"failures": 6, "next": 0}))  # the pause is over
        self.assertEqual(agent_gate.scan_jobs(settings, sweep=True), [])
        self.assertEqual(os.listdir(settings.jobs), [], "the retry state goes with its run")
        # An older gate that removes the run leaves its retry state to the next sweep.
        (settings.jobs / "0123456789abcdef.retry").write_text("{}")
        agent_gate.scan_jobs(settings, sweep=True)
        self.assertEqual(os.listdir(settings.jobs), [])

    def test_retries_stay_capped_however_long_a_survivor_lasts(self):
        self.ensure_state()
        settings = self.settings()
        record = {"id": "00000000cafef00d", "k": 1, "pid": 1, "start": 1, "admitted": time.time()}
        fd = agent_gate.publish_locked(settings.jobs, "00000000cafef00d.lock", record)
        self.addCleanup(os.close, fd)
        retry = settings.jobs / "00000000cafef00d.retry"
        retry.write_text(json.dumps({"failures": 5000, "next": 0}))
        self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [1])
        state = json.loads(retry.read_text())
        self.assertEqual(state["failures"], 5001)
        self.assertAlmostEqual(state["next"] - time.time(), agent_gate.ORPHAN_RETRY_SECONDS, delta=2)

    def test_retry_state_that_cannot_be_kept_leaves_attempts_unpaced(self):
        self.ensure_state()
        settings = self.settings()
        ended = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        record = {"id": "00000000f011f011", "k": 1, "pid": int(ended.stdout), "start": 1, "admitted": time.time()}
        fd = agent_gate.publish_locked(settings.jobs, "00000000f011f011.lock", record)
        real_open = os.open

        def full_disk(*_args: object) -> None:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        def no_retry_file(path: object, *args: object) -> int:
            if str(path).endswith(".retry"):
                full_disk()
            return real_open(path, *args)  # type: ignore[arg-type]

        try:
            with unittest.mock.patch.object(agent_gate.os, "pwrite", full_disk):
                self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [1])
            (settings.jobs / "00000000f011f011.retry").unlink()
            with unittest.mock.patch.object(agent_gate.os, "open", no_retry_file):
                self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [1])
                self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [1])
        finally:
            os.close(fd)
        self.assertEqual(len(self.events("orphan-survives")), 3, "every pass tried")
        with unittest.mock.patch.object(agent_gate.os, "open", no_retry_file):
            self.assertEqual(agent_gate.scan_jobs(settings, sweep=True), [])
        self.assertEqual(os.listdir(settings.jobs), [])

    def test_a_pause_beyond_the_longest_counts_as_set_by_a_clock_since_set_back(self):
        self.ensure_state()
        settings = self.settings()
        ended = subprocess.run([PY, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
        record = {"id": "00000000c10cc0de", "k": 1, "pid": int(ended.stdout), "start": 1, "admitted": time.time()}
        os.close(agent_gate.publish_locked(settings.jobs, "00000000c10cc0de.lock", record))
        retry = settings.jobs / "00000000c10cc0de.retry"
        retry.write_text(json.dumps({"failures": 3, "next": time.time() + 10_000}))
        self.assertEqual(agent_gate.scan_jobs(settings, sweep=True), [])
        self.assertEqual([event["event"] for event in self.events("orphan-stopped")], ["orphan-stopped"])
        self.assertEqual(os.listdir(settings.jobs), [])
        # A pause within the longest still holds.
        os.close(agent_gate.publish_locked(settings.jobs, "00000000c10cc0de.lock", record))
        retry.write_text(json.dumps({"failures": 3, "next": time.time() + agent_gate.ORPHAN_RETRY_SECONDS - 5}))
        self.assertEqual([job["k"] for job in agent_gate.scan_jobs(settings, sweep=True)], [1])
        self.assertEqual(len(self.events("orphan-stopped")), 1)

    def test_cleanup_signals_only_the_process_it_found(self):
        child = self.sleeper()
        start = gate_macos.process_info(child.pid)[1]
        # The pid now names a process that started at another time: a recycled pid.
        self.assertEqual(agent_gate.stop_processes({child.pid: start - 100}, lambda: {child.pid: start - 100}), 0)
        self.assertIsNone(child.poll())
        self.assertEqual(agent_gate.stop_processes({child.pid: start}, lambda: {}), 1)
        self.assertIsNotNone(child.wait(10))

    def test_only_writers_started_after_admission_hold_a_job_file(self):
        path = self.tmp / "job.lock"
        path.write_text("{}")
        writer = subprocess.Popen(
            [PY, "-c", f"import os, time; fd = os.open({str(path)!r}, os.O_RDWR); time.sleep(60)"])
        reader = subprocess.Popen(["/usr/bin/tail", "-f", str(path)], stdout=subprocess.DEVNULL)
        for proc in (writer, reader):
            self.addCleanup(proc.wait)
            self.addCleanup(proc.kill)
        self.wait_for(lambda: writer.pid in (agent_gate.lock_holders(path, 0) or {}), "lsof to see the writer")
        holders = agent_gate.lock_holders(path, 0) or {}
        self.assertNotIn(reader.pid, holders)
        self.assertEqual(agent_gate.lock_holders(path, time.time() + 100), {})

    def test_runner_limits_follow_the_tokens(self):
        probe = "import json, os; print(json.dumps({k: v for k, v in os.environ.items() if k.startswith(('AGENT_GATE_', 'VITEST_', 'PYTEST_'))}))"
        # Run in a small repository with an untracked file: the released event fingerprints its tree.
        repo = self.tmp / "repo"
        repo.mkdir()
        for args in (("init", "-q"), ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "t")):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        (repo / "untracked.txt").write_text("x")
        result = self.gate("-c", 3, "--", PY, "-c", probe, cwd=repo)
        values = json.loads(result.out)
        for name in ("AGENT_GATE_CPUS", *agent_gate.RUNNER_ENV):
            self.assertEqual(values[name], "3", name)
        self.assertRegex(values["AGENT_GATE_RUN"], r"^[0-9a-f]{16}$")
        released = self.events("released")[-1]
        self.assertEqual(released["k"], 3)
        self.assertRegex(released["tree"] or "", r"^[0-9a-f]{16}$", "the working tree was not fingerprinted")


class UsageTest(GateCase):
    def test_a_run_records_the_cpu_its_whole_tree_used(self):
        # Two grandchildren each spin one CPU for 0.8 s; the gate reaps only the command.
        spin = "import time; end = time.process_time() + 0.8\nwhile time.process_time() < end: pass"
        tree = (
            "import subprocess, sys; "
            f"procs = [subprocess.Popen([sys.executable, '-c', {spin!r}]) for _ in range(2)]; "
            "[proc.wait() for proc in procs]"
        )
        result = self.gate("-c", 2, "-l", "spin", "--", PY, "-c", tree)
        self.assertEqual(result.code, 0, result.err)
        released = self.events("released")[-1]
        self.assertGreaterEqual(released["cpu_s"], 1.5, released)
        self.assertAlmostEqual(released["cpus"], released["cpu_s"] / released["held_s"], delta=0.02)
        self.assertGreater(released["children_maxrss_mib"], 0)
        # The run's two CPUs are inside the host's busy share; the rest is everyone else's.
        self.assertGreater(released["host_busy"], 0)
        self.assertLessEqual(released["host_busy"], 1)
        self.assertGreaterEqual(released["others_cpus"], 0)

    def test_an_idle_run_records_almost_no_cpu(self):
        # The gate's own probes (ps, git, sysctl) run while it waits and are not the command's.
        result = self.gate("-l", "idle", "--", PY, "-c", "import time; time.sleep(1.2)")
        self.assertEqual(result.code, 0, result.err)
        self.assertLess(self.events("released")[-1]["cpus"], 0.25)

    def write_events(self, *events: Dict[str, object]) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        with open(self.state / "events.jsonl", "a") as log:
            for event in events:
                log.write(json.dumps({"event": "released", "ts": time.time(), **event}) + "\n")

    def usage(self, *args: str) -> List[Dict[str, object]]:
        result = subprocess.run(
            [PY, str(GATE), "usage", "--json", *args], env=self.env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_usage_sets_each_kind_of_command_against_its_claim(self):
        self.write_events(
            *({"label": f"web: pnpm exec vitest run src/a{i}.test.ts src/b.test.ts", "k": 4,
               "cpus": 0.9 + i / 10, "held_s": 10, "children_maxrss_mib": 300} for i in range(5)),
            {"label": "pytest: uv run pytest tests/x.py", "k": 1, "cpus": 1.7, "held_s": 30},
            {"label": "old: no measurement", "k": 4, "held_s": 30},
            {"label": "web: pnpm exec vitest run src/z.test.ts", "k": 4, "cpus": 3.0, "held_s": 5,
             "ts": time.time() - 10 * 3600},
        )
        with open(self.state / "events.jsonl", "ab") as log:
            log.write(b'null\n[1]\n{"event": "released", "label": "\xff"}\nnot json\n')
        rows = {row["kind"]: row for row in self.usage("--hours", "2")}
        self.assertEqual(set(rows), {"web: pnpm exec vitest run …", "pytest: uv run pytest …"})
        vitest = rows["web: pnpm exec vitest run …"]
        self.assertEqual((vitest["runs"], vitest["k"], vitest["fits"]), (5, 4, 1))
        self.assertEqual((vitest["cpus_p50"], vitest["cpus_p90"], vitest["token_s"]), (1.1, 1.3, 200))
        self.assertEqual(rows["pytest: uv run pytest …"]["fits"], 2, "a claim below its use shows too")
        self.assertEqual([row["kind"] for row in self.usage("--match", "pytest")], ["pytest: uv run pytest …"])

    def test_gates_that_find_the_log_full_at_once_lose_no_line(self):
        self.state.mkdir(mode=0o700)
        events = self.state / "events.jsonl"
        events.write_text("".join(json.dumps({"event": "old", "n": n}) + "\n" for n in range(50)))
        limit = events.stat().st_size - 1
        other = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(ROOT / "lib")!r})
            import agent_gate
            agent_gate.LOG_LIMIT_BYTES = {limit}
            agent_gate.log_event(agent_gate.Settings({{"AGENT_GATE_DIR": {str(self.state)!r}}}), {{"event": "other"}})
            """
        )
        replace, others = os.replace, []

        def racing(source: object, target: object) -> None:
            # Another gate logs, start to end, while this one rotates: it too found the log full.
            if not others:
                others.append(subprocess.Popen([PY, "-c", other]))
                others[0].wait(10)
            replace(source, target)

        with unittest.mock.patch.object(agent_gate, "LOG_LIMIT_BYTES", limit), \
                unittest.mock.patch.object(agent_gate.os, "replace", racing):
            agent_gate.log_event(self.settings(), {"event": "this"})
        self.assertEqual([other.returncode for other in others], [0], "the other gate logged during the rotation")
        logged = []
        for path in (events.with_name("events.jsonl.1"), events):
            with contextlib.suppress(FileNotFoundError):
                logged += [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(sorted(event["n"] for event in logged if event["event"] == "old"), list(range(50)),
                         "the full log was moved over by a fresh one")
        moved = [json.loads(line)["event"] for line in events.with_name("events.jsonl.1").read_text().splitlines()]
        self.assertEqual(moved[50:], ["other", "this"], "both lines end the log that was full")
        self.assertFalse(events.exists())

    def test_a_gate_that_takes_the_rotation_late_leaves_a_fresh_log_in_place(self):
        self.state.mkdir(mode=0o700)
        events = self.state / "events.jsonl"
        events.write_text("".join(json.dumps({"event": "old", "n": n}) + "\n" for n in range(50)))
        limit = events.stat().st_size - 1
        # A gate that found the log full and was then put off.
        late = os.open(events, os.O_WRONLY | os.O_APPEND)
        self.addCleanup(os.close, late)
        with unittest.mock.patch.object(agent_gate, "LOG_LIMIT_BYTES", limit):
            agent_gate.log_event(self.settings(), {"event": "moved"})
            agent_gate.log_event(self.settings(), {"event": "fresh"})
        agent_gate._rotate(events, late)
        moved = [json.loads(line)["event"] for line in events.with_name("events.jsonl.1").read_text().splitlines()]
        self.assertEqual(moved, ["old"] * 50 + ["moved"], "a fresh log was moved over the full one")
        self.assertEqual([json.loads(line)["event"] for line in events.read_text().splitlines()], ["fresh"])

    def test_the_window_must_be_a_finite_number_of_at_least_0(self):
        self.write_events({"label": "build", "k": 4, "cpus": 1.0, "held_s": 10})
        for hours in ("nan", "inf", "-inf", "-1", "x"):
            with self.subTest(hours=hours):
                result = subprocess.run([PY, str(GATE), "usage", "--json", f"--hours={hours}"], env=self.env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, agent_gate.CONFIG_EXIT, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("--hours", result.stderr)
        self.assertEqual([row["kind"] for row in self.usage("--hours", "1")], ["build"])
        self.assertEqual(self.usage("--hours", "0"), [])

    def test_usage_weighs_runs_by_their_time_and_leaves_docker_runs_unfitted(self):
        self.write_events(
            *({"label": "build", "k": 4, "cpus": 0.5, "held_s": 1} for _ in range(9)),
            {"label": "build", "k": 4, "cpus": 3.6, "held_s": 100},
            {"label": "e2e", "k": 4, "cpus": 0.4, "held_s": 60, "docker_gib": 1.5},
        )
        self.write_events({"label": "build", "k": 4, "cpus": 0.5, "held_s": 1, "others_cpus": 0.2},
                          {"label": "build", "k": 4, "cpus": 3.6, "held_s": 100, "others_cpus": 5.4})
        rows = {row["kind"]: row for row in self.usage()}
        self.assertEqual((rows["build"]["cpus_p50"], rows["build"]["fits"]), (3.6, 4), "a long busy run dominates")
        self.assertEqual(rows["build"]["others_cpus_p50"], 5.4, "others' CPUs weigh runs by time too")
        self.assertIsNone(rows["e2e"]["others_cpus_p50"], "runs before host readings have none")
        self.assertIsNone(rows["e2e"]["fits"], "a container's CPU runs in the Docker VM, outside the tree")

    def test_host_busy_survives_counter_wraps_and_unreadable_readings(self):
        wrap = 2**32
        before = ((wrap - 10, 5, 0), (wrap - 30,), wrap)
        after = ((20, 15, 0), (10,), wrap)  # busy 30 + 10, idle 40
        self.assertEqual(gate_macos.host_busy(before, after), 0.5)
        self.assertIsNone(gate_macos.host_busy(None, after))
        self.assertIsNone(gate_macos.host_busy(after, after), "no ticks passed")
        reading = agent_gate.host_cpu_ticks()
        self.assertIsNotNone(reading, "the host's counters are readable here")


class ReportTest(GateCase):
    """`status` and `usage` report what they can read, and exit 78 on a state directory they cannot."""

    def report(self, *args: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        return subprocess.run([PY, str(GATE), *args], env=self.env if env is None else env, capture_output=True,
                              text=True, timeout=30)

    def test_an_unreadable_state_directory_exits_78_with_a_one_line_reason(self):
        if os.geteuid() == 0:
            self.skipTest("root reads a mode 000 directory")
        as_file = self.tmp / "a-file"
        as_file.write_text("")
        locked = self.tmp / "locked"
        locked.mkdir(mode=0o700)
        (locked / "jobs").mkdir(mode=0o700)
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o700)
        for kind, path in (("a file", as_file), ("mode 000", locked)):
            env = dict(self.env, AGENT_GATE_DIR=str(path))
            for args in (["status"], ["status", "--json"], ["usage"], ["usage", "--json"]):
                with self.subTest(kind=kind, args=args):
                    result = self.report(*args, env=env)
                    self.assertEqual(result.returncode, agent_gate.CONFIG_EXIT, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertRegex(result.stderr, r"^agent-gate: unusable: [^\n]+\n$")
            with self.subTest(kind=kind, reader="gate_host"):
                # gate-host doctor reads the exit code and shows the reason on one line of its report.
                host = gate_host.Host(dict(env, GATE_HOST_ROOT=str(ROOT)))
                host.check_switch()
                [(state, subject)] = host.lines
                self.assertEqual(state, "broken")
                self.assertRegex(subject, r"^agent-gate status exited 78: agent-gate: unusable: [^\n]+$")

    def test_a_missing_or_open_state_directory_still_reports(self):
        # Nothing has run yet, or a run would refuse the directory's mode: both still report.
        missing = self.tmp / "missing"
        loose = self.tmp / "loose"
        loose.mkdir(mode=0o755)
        loose.chmod(0o755)
        for kind, path in (("missing", missing), ("mode 0755", loose)):
            env = dict(self.env, AGENT_GATE_DIR=str(path))
            for args in (["status", "--json"], ["usage", "--json"]):
                with self.subTest(kind=kind, args=args):
                    result = self.report(*args, env=env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    json.loads(result.stdout)
        result = self.gate("--", PY, "-c", "pass", env=dict(self.env, AGENT_GATE_DIR=str(loose)))
        self.assertEqual(result.code, agent_gate.CONFIG_EXIT, "a run refuses what a report reads")


class DockerTest(GateCase):
    def setUp(self) -> None:
        super().setUp()
        self.docker = self.tmp / "docker"
        self.docker.mkdir()
        program = self.docker / "docker"
        program.write_text(f"#!{PY}\n" + FAKE_DOCKER)
        program.chmod(0o755)
        self.env.update(FAKE_DOCKER_DIR=str(self.docker), PATH=f"{self.docker}{os.pathsep}{self.env.get('PATH', '')}")
        self.set_vm(3, 5)

    def set_vm(self, available: float, total: float) -> None:
        (self.docker / "vm").write_text(f"{available} {total}")

    def busy(self, gib: float) -> None:
        """A running stack that uses `gib` of the VM, which stopping it would give back."""
        (self.docker / "ps").write_text("b1\tbusy\tbusy-db-1\n")
        (self.docker / "stats").write_text(f"b1\t{gib}GiB / 4GiB\n")

    def docker_calls(self, caller: Optional[int] = None) -> List[str]:
        """The docker calls made so far, without their callers' pids, or only `caller`'s."""
        try:
            lines = (self.docker / "calls").read_text().splitlines()
        except FileNotFoundError:
            return []
        calls = [line.split(" ", 1) for line in lines]
        return [call for pid, call in calls if caller is None or int(pid) == caller]

    def reserved(self) -> List[float]:
        return [holder["docker_gib"] for holder in agent_gate.docker_holders(self.settings())]

    def test_docker_runs_reserve_what_they_declare(self):
        first = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("first"), name="first")
        self.command_pid("first")
        self.assertEqual(self.reserved(), [2])
        # The VM still shows 3 GiB available, but 2 of them are the first run's.
        second = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("second", "sleep:0"), name="second")
        self.wait_for(lambda: "Docker VM memory" in self.err("second"), "the second run to queue")
        self.assertIn("2.00 GiB of it reserved by 1 earlier run(s); 2 GiB needed", self.err("second"))
        self.assertIn("largest users: no running containers", self.err("second"))
        time.sleep(1)
        self.assertFalse(self.started("second"))
        self.end(first)
        result = self.finish(second)
        self.assertEqual(result.code, 0, result.err)
        self.assertIn("admitted", result.err)
        self.assertEqual(os.listdir(self.state / "docker/jobs"), [])
        self.assertEqual(os.listdir(self.state / "docker/waiters"), [])
        released = self.events("released")
        self.assertEqual([event["docker_gib"] for event in released], [2, 2])
        self.assertGreater(released[-1]["docker_waited_s"], 1)

    def test_off_starts_requests_queued_for_the_vm(self):
        first = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("first"), name="first")
        self.command_pid("first")
        head = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("head"), name="head")
        self.wait_for(lambda: "Docker VM memory" in self.err("head"), "the head to wait for the VM")
        behind = self.start("--docker-gib", 1, "-c", 1, "--", *self.held("behind"), name="behind")
        self.wait_for(lambda: self.queued(docker=True) == 2, "a request to queue behind it")
        switched = time.monotonic()
        self.assertEqual(self.gate("off").code, 0)
        self.command_pid("head")
        self.command_pid("behind")
        self.assertLess(time.monotonic() - switched, 2.0, "both start once the gate is off")
        self.assertEqual(self.queued(docker=True), 0)
        self.assertEqual(self.reserved(), [2], "they reserve nothing")
        self.end(first, head, behind)

    def test_waiting_for_the_vm_holds_no_tokens(self):
        self.set_vm(1, 5)
        (self.docker / "ps").write_text("c1\tsample\tsample-db-1\nc2\t\tloose\n")
        (self.docker / "stats").write_text("c1\t1.5GiB / 2GiB\nc2\t100MiB / 1GiB\n")
        waiting = self.start("--docker-gib", 2, "-c", 4, "--", *self.held("waiting", "sleep:0"), name="waiting")
        self.wait_for(lambda: self.queued(docker=True) == 1 and "Docker VM memory" in self.err("waiting"),
                      "the Docker run to queue")
        self.assertIn("largest users: sample 1.50 GiB, loose 0.10 GiB", self.err("waiting"))
        # The whole pool is free for runs that need no Docker memory.
        result = self.gate("--no-wait", "-c", 4, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        report = self.status()
        self.assertEqual(len(report["docker"]["waiting"]), 1)
        self.assertEqual(report["docker"]["vm"]["available_gib"], 1)
        text = self.status_text()
        self.assertIn("Docker VM: 1 of 5 GiB available", text)
        self.assertIn("queued #1 2 GiB", text)
        self.set_vm(2.5, 5)
        result = self.finish(waiting)
        self.assertEqual(result.code, 0, result.err)
        self.assertTrue(self.started("waiting"))

    def test_a_dead_docker_waiter_is_logged_as_abandoned(self):
        self.set_vm(1, 5)
        waiting = self.start("--docker-gib", 2, "-c", 1, "-l", "vm", "--", *self.held("never", "sleep:0"),
                             name="waiting")
        self.wait_for(lambda: self.queued(docker=True) == 1, "the Docker run to queue")
        waiting.kill()
        waiting.wait()
        self.set_vm(3, 5)
        # Last in line, its file waits for the next Docker request to collect it.
        self.assertEqual(self.events("abandoned"), [])
        result = self.gate("--docker-gib", 2, "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        [event] = self.events("abandoned")
        self.assertEqual((event["queue"], event["label"], event["docker_gib"]), ("docker", "vm", 2))
        self.assertFalse(self.started("never"))

    def test_an_unreadable_vm_allows_one_docker_run_at_a_time(self):
        (self.docker / "down").touch()
        first = self.start("--docker-gib", 1, "-c", 1, "--", *self.held("first"), name="first")
        self.command_pid("first")
        second = self.start("--docker-gib", 1, "-c", 1, "--", *self.held("second", "sleep:0"), name="second")
        self.wait_for(lambda: "unreadable" in self.err("second"), "the second run to wait")
        self.assertIn("Cannot connect to the Docker daemon", self.err("second"))
        self.assertIn("one Docker run at a time", self.err("second"))
        self.assertEqual(self.gate("--no-wait", "-c", 1, "--", PY, "-c", "pass").code, 0)
        self.end(first)
        self.assertEqual(self.finish(second).code, 0)

    def test_impossible_or_malformed_claims_are_configuration_errors(self):
        for args in (["--docker-gib", 6], ["--docker-gib", -1], ["--docker-gib", "nan"], ["--docker-frees", "x"]):
            with self.subTest(args):
                result = self.gate("--no-wait", *args, "-c", 1, "--", *self.held("never", "sleep:0"))
                self.assertEqual(result.code, agent_gate.CONFIG_EXIT, result.err)
        self.assertIn("exceeds the Docker VM's 5.00 GiB", self.err("run0"))
        self.assertFalse(self.started("never"))
        self.assertEqual(os.listdir(self.state / "docker/jobs"), [])

    def test_wait_limits_cover_both_queues(self):
        self.set_vm(1, 5)
        self.busy(2)
        result = self.gate("--no-wait", "--docker-gib", 2, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertLess(result.elapsed, 4)
        self.assertIn("1.00 GiB available; 2 GiB needed", result.err)
        result = self.gate("--wait", 1, "--docker-gib", 2, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertTrue(0.9 <= result.elapsed < 4, result.elapsed)
        # Admitted to the VM, then out of time in the token queue: the reservation goes back.
        self.set_vm(3, 5)
        blocker = self.holding("blocker", 4)
        result = self.gate("--wait", 1, "--docker-gib", 2, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT)
        self.assertIn("tokens in use", result.err)
        self.assertFalse(self.started("never"))
        self.assertEqual(os.listdir(self.state / "docker/jobs"), [])
        self.assertEqual(os.listdir(self.state / "docker/waiters"), [])
        self.assertEqual([event["docker_gib"] for event in self.events("not-admitted")], [2, 2, 2])
        self.end(blocker)

    def test_a_compose_project_the_command_stops_counts_as_available(self):
        self.set_vm(1, 5)
        (self.docker / "ps").write_text("c1\tmine\tmine-api-1\nc2\tmine\tmine-db-1\nc3\tother\tother-db-1\n"
                                        "c4\t\tother\n")
        (self.docker / "stats").write_text("c1\t1GiB / 2GiB\nc2\t512MiB / 1GiB\nc3\t600MiB / 1GiB\n"
                                           "c4\t900MiB / 1GiB\n")
        result = self.gate("--no-wait", "--docker-gib", 2, "--docker-frees", "other", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, "a container merely named like it is not freed")
        self.assertIn("other frees 0.59 GiB", result.err)
        result = self.gate("--no-wait", "--docker-gib", 2, "--docker-frees", "mine", "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(self.events("admitted")[-1]["docker_gib"], 2)

    def test_the_command_inherits_the_reservation(self):
        proc = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("orphan"), name="orphan")
        pid = self.command_pid("orphan")
        proc.kill()
        proc.wait()
        self.assertEqual(self.reserved(), [2], "a killed gate's command keeps its reservation")
        self.release("orphan")
        self.wait_for(lambda: self.gone(pid), "the command to finish")
        self.assertEqual(self.reserved(), [])
        result = self.gate("--no-wait", "--docker-gib", 3, "-c", 1, "--", PY, "-c", "pass")
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(os.listdir(self.state / "docker/jobs"), [])

    def test_a_nested_gate_runs_within_the_outer_reservation(self):
        script = self.gate_script(
            """
            gate += ["--no-wait", "-c", "1"]
            for gib in ("1", "2"):
                result = subprocess.run(gate + ["--docker-gib", gib, "--", sys.executable, "-c", "pass"],
                                        capture_output=True, text=True)
                print(gib, result.returncode, "declare it" in result.stderr, flush=True)
            """
        )
        result = self.gate("--docker-gib", 1, "-c", 2, "--", PY, "-c", script)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.out.splitlines(), ["1 0 False", "2 0 True"])
        self.assertEqual([event["docker_gib"] for event in self.events("nested")], [1, 2])
        self.assertEqual(sum(call.startswith("run ") for call in self.docker_calls()), 1, "only the outer run probes")

    def test_the_docker_queue_is_fifo_and_only_its_head_probes(self):
        self.set_vm(1, 5)
        self.busy(2)
        head = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("head", "sleep:0"), name="head")
        self.wait_for(lambda: "Docker VM memory" in self.err("head"), "the head to queue")
        # The VM has room for this one, but it may not pass the head.
        later = self.start("--docker-gib", 0.5, "-c", 1, "--", *self.held("later", "sleep:0"), name="later")
        self.wait_for(lambda: self.queued(docker=True) == 2, "the later run to queue")
        self.wait_for(lambda: "position 2" in self.err("later"), "the later run's position")
        time.sleep(agent_gate.DOCKER_POLL_SECONDS + 1)
        self.assertFalse(self.started("later"))
        self.assertEqual(self.docker_calls(later.pid), [], "only the head reads the VM")
        self.assertGreaterEqual(sum(call.startswith("run ") for call in self.docker_calls(head.pid)), 2)
        self.set_vm(2.5, 5)
        self.assertEqual(self.finish(head).code, 0)
        self.assertEqual(self.finish(later).code, 0)
        self.assertLess(self.times("start")["head"], self.times("start")["later"])

    def test_a_claim_the_vm_could_never_offer_is_refused(self):
        self.set_vm(1, 5)
        (self.docker / "ps").write_text("c1\tstack\tstack-db-1\n")
        (self.docker / "stats").write_text("c1\t1GiB / 2GiB\n")
        # One short survey may only see a build or a stopping container take memory: it waits.
        result = self.gate("--no-wait", "--docker-gib", 2.5, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("with every container stopped it could offer 2.00 GiB now", result.err)
        # Short at every survey for the whole window, it is refused as a configuration error.
        result = self.gate("--docker-gib", 2.5, "-c", 1, "--", *self.held("never", "sleep:0"),
                           env=dict(self.env, AGENT_GATE_DOCKER_CEILING_SECONDS="0"))
        self.assertEqual(result.code, agent_gate.CONFIG_EXIT, result.err)
        self.assertIn("exceeds what the Docker VM could offer", result.err)
        self.assertIn("at most 2.00 GiB", result.err)
        # Stopping that stack would make room for this one, so it waits.
        result = self.gate("--no-wait", "--docker-gib", 1.5, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("largest users: stack 1.00 GiB", result.err)
        self.assertFalse(self.started("never"))
        self.assertEqual(os.listdir(self.state / "docker/waiters"), [])
        reasons = [event["reason"].startswith("unusable:") for event in self.events("not-admitted")]
        self.assertEqual(reasons, [False, True, False], "every departure is logged")

    def test_an_unread_credit_is_named_in_the_refusal(self):
        self.set_vm(1, 5)
        (self.docker / "ps").write_text("c1\tstack\tstack-db-1\n")
        (self.docker / "stats").write_text("c1\t1GiB / 2GiB\n")
        (self.docker / "fail-stats").touch()
        result = self.gate("--no-wait", "--docker-gib", 1.5, "--docker-frees", "stack", "-c", 1, "--",
                           *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("nothing credited for stack", result.err)
        (self.docker / "fail-stats").unlink()
        result = self.gate("--no-wait", "--docker-gib", 1.5, "--docker-frees", "stack", "-c", 1, "--",
                           *self.held("now", "sleep:0"))
        self.assertEqual(result.code, 0, result.err)

    def test_a_docker_that_does_not_answer_keeps_to_the_wait_limit(self):
        first = self.start("--docker-gib", 1, "-c", 1, "--", *self.held("first"), name="first")
        self.command_pid("first")
        (self.docker / "hang").touch()
        result = self.gate("--no-wait", "--docker-gib", 1, "-c", 1, "--", *self.held("never", "sleep:0"))
        self.assertEqual(result.code, agent_gate.NOT_ADMITTED_EXIT, result.err)
        self.assertIn("unreadable", result.err)
        self.assertLess(result.elapsed, agent_gate.DOCKER_MIN_SECONDS + 4)
        self.end(first)

    def test_an_interrupt_gives_the_reservation_back(self):
        self.set_vm(1, 5)
        self.busy(2)
        waiting = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("never", "sleep:0"), name="waiting")
        self.wait_for(lambda: "Docker VM memory" in self.err("waiting"), "the run to queue for the VM")
        waiting.send_signal(signal.SIGINT)
        self.assertEqual(self.finish(waiting).code, 130)
        self.set_vm(3, 5)
        blocker = self.holding("blocker", 4)
        tokens = self.start("--docker-gib", 2, "-c", 1, "--", *self.held("never", "sleep:0"), name="tokens")
        self.wait_for(lambda: self.reserved() == [2] and self.queued() == 1, "the run to wait for tokens")
        tokens.send_signal(signal.SIGINT)
        self.assertEqual(self.finish(tokens).code, 130)
        self.assertFalse(self.started("never"))
        self.assertEqual(self.reserved(), [])
        self.assertEqual(os.listdir(self.state / "docker/jobs"), [])
        self.assertEqual(os.listdir(self.state / "docker/waiters"), [])
        self.end(blocker)

    def test_the_probe_runs_an_image_already_present(self):
        settings = agent_gate.Settings(dict(self.env, AGENT_GATE_DOCKER_PROBE_IMAGES="busybox, redis:7"))
        self.assertEqual(settings.probe_images, ("busybox:latest", "redis:7"))
        environ = {"PATH": self.env["PATH"], "FAKE_DOCKER_DIR": str(self.docker)}
        with unittest.mock.patch.dict(os.environ, environ):
            (self.docker / "images").write_text("node:20-slim\nredis:7\n")
            self.assertEqual(agent_gate.docker_memory(settings), (3, 5))
            (self.docker / "images").write_text("node:20-slim\n")
            agent_gate.docker_memory(settings)
            (self.docker / "stats").write_text("a\t1.5GiB / 2GiB\nb\t512kB / 1GiB\nc\t1GB / 2GB\nd\t--\n")
            (self.docker / "ps").write_text("a\tp\ta-1\nb\tp\tb-1\nc\t\tloose\nd\tq\td-1\n")
            usage = agent_gate.by_project(agent_gate.containers())
        runs = [call for call in self.docker_calls() if call.startswith("run ")]
        self.assertEqual(runs, [
            "run --rm --pull=never --network=none --log-driver=none --entrypoint cat redis:7 /proc/meminfo",
            "run --rm --pull=missing --network=none --log-driver=none --entrypoint cat busybox:latest /proc/meminfo",
        ])
        self.assertEqual([(name, round(gib, 3)) for name, gib in usage], [("p", 1.5), ("loose", 0.931)])


if __name__ == "__main__":
    unittest.main()
