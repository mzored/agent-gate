"""PATH shims and the agent-verify front end; isolated fixtures never touch the live gate."""

from __future__ import annotations

import os
from pathlib import Path
import pty
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Sequence
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
PY = sys.executable
sys.path.insert(0, str(ROOT / "lib"))
from gate_host import TOOLS  # noqa: E402

# The invocation of each shimmed tool that runs the project's tests. A tool added to gate_host.TOOLS
# needs an entry here, and devshim has to classify it as heavy.
TEST_VERB = {
    "bun": ("bun", "test"),
    "make": ("make", "test"),
    "npm": ("npm", "test"),
    "npx": ("npx", "vitest", "run"),
    "playwright": ("playwright", "test"),
    "pnpm": ("pnpm", "test"),
    "poetry": ("poetry", "run", "pytest"),
    "pytest": ("pytest",),
    "uv": ("uv", "run", "pytest"),
    "vitest": ("vitest", "run"),
    "yarn": ("yarn", "test"),
}


class Fixture:
    """A checkout, a shim directory linked to this repository's devshim, and fake real tools."""

    def __init__(self, devshim: Path = BIN / "devshim"):
        self.tmp = Path(tempfile.mkdtemp(prefix="devshim-test-"))
        self.root = self.tmp / "checkout"
        (self.root / ".git").mkdir(parents=True)
        self.shims = self.tmp / "shims"
        self.real = self.tmp / "real"
        self.shims.mkdir()
        self.real.mkdir()
        for tool in TOOLS:
            (self.shims / tool).symlink_to(devshim)
            fake = self.real / tool
            fake.write_text(f'#!/bin/sh\necho "{tool} AGENT_GATE_CPUS=${{AGENT_GATE_CPUS:-none}}"\nexit 7\n')
            fake.chmod(0o755)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("AGENT_", "DEVSHIM_", "VITEST_", "PYTEST_XDIST")) and key != "CI"
        }
        self.env.update(
            PATH=f"{self.shims}:{self.real}:/usr/bin:/bin",
            AGENT_GATE_DIR=str(self.tmp / "state"),
            AGENT_GATE_MEMORY_FILE=str(self.tmp / "memory"),
            AGENT_GATE_TOKENS="4",
            # These tests map options onto whole-pool claims; test_agent_gate.ClaimTest covers the cap.
            AGENT_GATE_MAX_CLAIM="4",
            AGENT_GATE_RAMP_SECONDS="0",
            # A full pool refuses every claim here; test_agent_gate.BackfillTest covers the 1-token headroom.
            AGENT_GATE_SMALL_HEADROOM="0",
        )
        (self.tmp / "memory").write_text("1 80")

    def run(self, command: Sequence[str], explain: bool = True, tty: bool = False,
            **extra: str) -> subprocess.CompletedProcess:
        env = dict(self.env, **extra)
        if explain:
            env["DEVSHIM_EXPLAIN"] = "1"
        if not tty:
            return subprocess.run(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=60)
        leader, follower = pty.openpty()
        try:
            return subprocess.run(command, cwd=self.root, env=env, stdin=follower,
                                  capture_output=True, text=True, timeout=60)
        finally:
            os.close(leader)
            os.close(follower)

    def decision(self, command: Sequence[str], **kwargs) -> str:
        result = self.run(command, **kwargs)
        for word in result.stdout.split():
            if word.startswith("decision="):
                return word[len("decision="):]
        raise AssertionError(f"{command}: no decision in {result.stdout!r} {result.stderr!r}")

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class DevshimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        self.addCleanup(self.fixture.cleanup)

    def assertDecisions(self, expected: str, commands: Sequence[Sequence[str]], **kwargs) -> None:
        for command in commands:
            with self.subTest(command=" ".join(command)):
                self.assertEqual(self.fixture.decision(command, **kwargs), expected)

    def test_verification_that_fans_out_takes_the_gate(self):
        self.assertDecisions("gate", (
            ["npm", "test"],
            ["npm", "run", "test:e2e"],
            ["npm", "--prefix", "client", "run", "build"],
            ["pnpm", "asset:hero:build"],
            ["pnpm", "asset:hero:test-reproducibility"],
            ["pnpm", "asset:hero:verify"],
            ["pnpm", "asset:hero:check-schema"],
            ["pnpm", "run", "check"],
            ["make", "help", "assets:build-prod"],
            ["make", "-j", "4", "test"],
            ["uv", "run", "python", "-m", "pytest"],
            ["uv", "run", "/usr/bin/python3", "-m", "pytest"],
            ["poetry", "run", "python3", "-X", "dev", "-m", "unittest"],
            ["pytest", "-q"],
            ["vitest", "run"],
            ["vitest"],
            ["npx", "vitest", "related", "src/a.ts"],
            ["playwright", "test"],
        ))

    def test_short_single_process_tools_dev_servers_and_installs_run_directly(self):
        self.assertDecisions("direct", (
            ["npm", "run", "format:check"],
            ["npm", "run", "typecheck:tests"],
            ["pnpm", "check:types"],
            ["pnpm", "asset:hero:format-check"],
            ["pnpm", "lint"],
            ["npx", "tsc", "--noEmit"],
            ["uv", "run", "ruff", "check", "."],
            ["uv", "run", "mypy", "."],
            ["poetry", "run", "black", "--check", "."],
            ["uv", "run", "python", "script.py", "-m", "pytest"],
            ["uv", "run", "python", "-c", "pass", "-m", "pytest"],
            ["make", "-C", "tests", "lint"],
            ["pnpm", "dev"],
            ["npm", "install"],
            ["npm", "ci"],
            ["vitest", "--watch"],
            ["vitest", "watch"],
            ["playwright", "show-report"],
            ["playwright", "test", "--ui"],
        ))

    def test_a_bare_vitest_in_a_terminal_is_a_watcher(self):
        self.assertDecisions("direct", (["vitest"], ["pnpm", "exec", "vitest", "src/a.test.ts"]), tty=True)
        self.assertDecisions("gate", (["vitest", "run"], ["vitest", "--run"]), tty=True)
        self.assertDecisions("gate", (["vitest"],), tty=True, CI="1")

    def test_repositories_that_own_admission_run_their_entry_points_directly(self):
        (self.fixture.root / ".agent-admission").write_text("")
        self.assertDecisions("owned", (["npm", "test"], ["make", "test"], ["pytest"]))
        self.assertDecisions("gate", (["pnpm", "build"],))

    def test_legacy_vendored_admission_files_are_ignored(self):
        # Version-1 vendored copies (scripts/agent_admission.py, scripts/validation/bounded.py) once marked a
        # checkout as running its own admission. Only `.agent-admission` does now; the shims must not honour
        # them, so a checkout holding only these files takes the gate like any other. This does not cover
        # any project's current runner.
        validation = self.fixture.root / "scripts" / "validation"
        validation.mkdir(parents=True)
        runner = self.fixture.root / "scripts" / "verify"
        runner.write_text("#!/bin/sh\nexit 0\n")
        runner.chmod(0o755)
        for path in (self.fixture.root / "scripts" / "agent_admission.py", validation / "agent_admission.py",
                     validation / "bounded.py"):
            path.write_text("# a legacy vendored admission file\n")
        self.assertDecisions("gate", (["npm", "test"], ["pnpm", "verify"], ["make", "test"]))
        self.assertDecisions("direct", (["pnpm", "lint"],))

    def checkout(self, path: Path, owned: bool = False) -> Path:
        (path / ".git").mkdir(parents=True)
        if owned:
            (path / ".agent-admission").write_text("")
        return path

    def test_every_shimmed_tool_takes_the_gate_for_its_test_verb(self):
        self.assertEqual(sorted(TEST_VERB), sorted(TOOLS),
                         "name the test invocation of every tool in gate_host.TOOLS in TEST_VERB")
        self.assertDecisions("gate", [list(verb) for verb in TEST_VERB.values()])

    def test_hyphenated_watchers_run_directly(self):
        self.assertDecisions("direct", (
            ["pnpm", "test-watch"],
            ["pnpm", "watch-test"],
            ["npm", "run", "watch-test"],
            ["npm", "run", "test-watch"],
            ["yarn", "test-watch"],
            ["bun", "run", "test-watch"],
            ["pnpm", "test:watch:ui"],
            ["make", "test-watch"],
            ["make", "watch-test"],
        ))
        self.assertDecisions("gate", (["pnpm", "test:no-watch"], ["make", "test-no-watch"], ["pnpm", "watcher-test"]))

    def test_version_help_and_collection_flags_on_runners_run_directly(self):
        self.assertDecisions("direct", (
            ["pytest", "--version"],
            ["pytest", "-V"],
            ["pytest", "--help"],
            ["pytest", "-h"],
            ["pytest", "--collect-only", "tests"],
            ["pytest", "-q", "--co"],
            ["playwright", "--help"],
            ["playwright", "test", "--help"],
            ["playwright", "--version"],
            ["vitest", "--help"],
            ["vitest", "--version"],
            ["npx", "vitest", "--help"],
            ["uv", "run", "pytest", "--collect-only"],
        ))
        self.assertDecisions("gate", (
            ["pytest", "-k", "version"],
            ["pytest", "tests/help"],
            ["pytest", "--", "--help"],
            ["pytest", "-q"],
        ))

    def test_make_targets_get_the_same_dev_start_and_watch_exclusions_as_scripts(self):
        self.assertDecisions("direct", (
            ["make", "dev-build"],
            ["pnpm", "dev-build"],
            ["make", "dev"],
            ["make", "start"],
            ["make", "serve"],
            ["make", "up-test"],
            ["make", "logs-test"],
        ))
        self.assertDecisions("gate", (["make", "test", "dev-build"], ["make", "dev-build", "test"]))

    def test_a_bare_make_runs_directly(self):
        # Owner-visible call, kept: a bare make runs the Makefile's default target, which is `help` or `all`
        # in most projects and says nothing about verification.
        self.assertDecisions("direct", (["make"], ["make", "-j", "4"], ["make", "-s"]))

    def test_the_verb_is_found_behind_workspace_selectors_and_unknown_options(self):
        self.assertDecisions("gate", (
            ["yarn", "workspace", "web", "test"],
            ["yarn", "workspace", "web", "run", "test"],
            ["yarn", "workspaces", "foreach", "-A", "run", "test"],
            ["yarn", "workspaces", "foreach", "--all", "test"],
            ["yarn", "workspaces", "foreach", "-j", "4", "run", "build"],
            ["pnpm", "--reporter", "append-only", "test"],
            ["pnpm", "-r", "--workspace-concurrency", "2", "test"],
            ["pnpm", "-r", "test"],
            ["pnpm", "--no-such-option", "value", "run", "test"],
            ["pnpm", "recursive", "run", "test"],
            ["npm", "--no-such-option", "value", "test"],
            ["npm", "-w", "web", "run", "build"],
            ["bun", "--filter", "web", "test"],
            ["npm", "t"],
            ["pnpm", "tst"],
            ["npm", "it"],
            ["npm", "--silent", "t"],
        ))
        self.assertDecisions("direct", (
            ["yarn", "workspace", "web", "lint"],
            ["yarn", "workspace", "web", "dev"],
            ["yarn", "workspace", "test-utils", "lint"],
            ["yarn", "workspaces", "foreach", "-A", "run", "lint"],
            ["yarn", "workspaces", "info"],
            ["pnpm", "--reporter", "append-only", "lint"],
            ["pnpm", "-r", "--workspace-concurrency", "2", "lint"],
            ["pnpm", "--silent", "lint", "tests"],
            ["pnpm", "--filter", "test-utils", "lint"],
            ["npm", "--no-such-option", "value", "run", "lint"],
            ["npm", "run", "t"],
            # An unknown option does not turn an install into its package's name.
            ["npm", "--silent", "install", "test-utils"],
            ["pnpm", "-w", "add", "-D", "vitest"],
            ["pnpm", "-r", "add", "vitest"],
            ["poetry", "install", "--with", "test"],
            ["uv", "run", "--frozen", "python", "script.py", "test"],
        ))

    def test_npx_runs_the_first_word_that_is_not_an_option(self):
        self.assertDecisions("gate", (
            ["npx", "-y", "vitest", "run"],
            ["npx", "--package", "vitest", "vitest", "run"],
            ["npx", "--no-such-option", "vitest", "run"],
        ))
        self.assertDecisions("direct", (
            ["npx", "--yes", "create-foo", "test"],
            ["npx", "-p", "typescript", "tsc"],
        ))

    def test_uv_and_poetry_verbs_are_found_behind_options_and_tool_run(self):
        self.assertDecisions("gate", (
            ["uv", "run", "--package", "foo", "pytest"],
            ["uv", "run", "--package", "foo", "python", "-m", "pytest"],
            ["uv", "run", "--no-such-option", "value", "pytest"],
            ["uv", "run", "--frozen", "pytest"],
            ["uv", "tool", "run", "pytest"],
            ["uv", "tool", "run", "--from", "pytest-cov", "pytest"],
            ["poetry", "-P", "sub", "run", "pytest"],
            ["poetry", "--project", "sub", "run", "pytest"],
            ["poetry", "-C", "sub", "run", "pytest"],
        ))
        self.assertDecisions("direct", (
            ["uv", "run", "--package", "foo", "ruff", "check", "."],
            ["uv", "run", "--package", "foo", "python", "script.py", "-m", "pytest"],
            ["uv", "tool", "install", "ruff"],
            ["uv", "tool", "list"],
            ["uv", "run", "--package", "test-utils", "python", "script.py"],
            ["poetry", "-P", "sub", "run", "ruff", "check"],
        ))

    def test_make_directory_options_pick_the_repository_that_declares_admission(self):
        other = self.checkout(self.fixture.tmp / "other", owned=True)
        nested = self.checkout(self.fixture.root / "nested", owned=True)
        self.assertDecisions("owned", (
            ["make", "-C", str(other), "test"],
            ["make", f"-C{other}", "test"],
            ["make", "-sC", str(other), "test"],
            ["make", f"-sC{other}", "test"],
            ["make", "-ksC", str(other), "test"],
            ["make", "-dkC", str(other), "test"],
            ["make", "--directory", str(other), "test"],
            ["make", f"--directory={other}", "test"],
            ["make", "-sC", "nested", "test"],
            ["make", "-s", "-C", "nested", "-j", "2", "test"],
            ["make", "-sj", "2", "-C", "nested", "test"],
        ))
        self.assertDecisions("gate", (
            ["make", "-sf", "Makefile.alt", "test"],
            ["make", "-sj", "4", "test"],
            ["make", "-sj", "test"],
            ["make", "-sOtarget", "test"],
        ))
        # Without the marker where make runs, the gate stays.
        plain = self.checkout(self.fixture.tmp / "plain")
        self.assertEqual(self.fixture.decision(["make", "-sC", str(plain), "test"]), "gate")
        self.assertEqual(nested.name, "nested")
        # With the marker only in the starting directory, -C moves the check away from it.
        (self.fixture.root / ".agent-admission").write_text("")
        self.assertDecisions("gate", (
            ["make", "-sC", str(plain), "test"],
            ["make", f"-sC{plain}", "test"],
            ["make", "-C", str(plain), "-s", "test"],
        ))
        self.assertDecisions("owned", (["make", "test"], ["make", "-s", "test"], ["make", "-f", "Makefile.alt", "test"]))

    def test_launcher_directory_options_pick_the_repository_that_declares_admission(self):
        other = self.checkout(self.fixture.tmp / "other", owned=True)
        plain = self.checkout(self.fixture.tmp / "plain")
        self.assertDecisions("owned", (
            ["poetry", "-P", str(other), "run", "pytest"],
            ["poetry", "--project", str(other), "run", "pytest"],
            ["poetry", "-C", str(other), "run", "pytest"],
            ["uv", "run", "--directory", str(other), "pytest"],
            ["uv", "--project", str(other), "run", "pytest"],
            ["uv", "run", f"--project={other}", "pytest"],
            ["pnpm", "--filter", "web", "-C", str(other), "test"],
            ["pnpm", "--no-such-option", "value", "--dir", str(other), "test"],
            ["pnpm", "--reporter", "append-only", f"--dir={other}", "test"],
            ["npm", "--prefix", str(other), "run", "test"],
            ["npm", "run", "--prefix", str(other), "test"],
        ))
        (self.fixture.root / ".agent-admission").write_text("")
        # uv's -C is --config-setting and its -P is --upgrade-package: neither names a directory.
        self.assertDecisions("owned", (
            ["uv", "run", "-C", f"key={plain}", "pytest"],
            ["uv", "run", "-P", str(plain), "pytest"],
        ))
        self.assertDecisions("gate", (["poetry", "-P", str(plain), "run", "pytest"], ["uv", "run", "--directory", str(plain), "pytest"]))

    def test_test_uis_servers_and_debuggers_run_directly(self):
        self.assertDecisions("direct", (
            ["pnpm", "test:ui"],
            ["pnpm", "e2e:ui"],
            ["pnpm", "test:debug"],
            ["npm", "run", "test:serve"],
            ["pnpm", "test:e2e:ui"],
            ["pnpm", "test:vitest:ui"],
            ["pnpm", "test:watch:ui"],
            ["yarn", "test-ui"],
        ))

    def test_names_that_only_mention_a_ui_server_or_debugger_are_still_suites(self):
        # The interactive token has to end the name, right after a token for tests, and no ci, run, headless,
        # unit or no token may say the run is one-shot. A false direct overloads the host.
        self.assertDecisions("gate", tuple(["pnpm", name] for name in (
            "test:ui:ci", "ui:test", "ui-tests", "test:ui:unit", "test:unit:ui", "test:ui-components",
            "test:web-ui", "e2e:debug-ci", "test:serve:ci", "test-build-ui", "build:ui:test", "cov:ui",
            "bench:ui", "coverage:serve", "test:serve-static", "test:ui:run", "test:run:ui", "test:ui:headless",
            "test:no-ui", "build:ui", "build:debug", "test:uix",
        )) + (
            # make has no interactive mode for a target: a test-ui target is a suite.
            ["make", "test-ui"], ["make", "ui-test"], ["make", "serve-test"], ["make", "debug-test"],
            ["make", "e2e-debug-ci"], ["make", "test-ui-build"], ["make", "e2e-ui"], ["make", "test-serve"],
        ))

    def test_watch_in_any_position_is_long_lived_unless_the_run_finishes(self):
        self.assertDecisions("direct", tuple(["pnpm", name] for name in (
            "test:watch", "watch", "watch:test", "test:watch:e2e", "test:watch:coverage", "test:watch:unit",
            "test:watch:changed", "test:watch-all", "build:watch:types", "docs:watch:build",
        )) + (["make", "test-watch-unit"], ["make", "test-watch"], ["make", "watch-test"]))
        # A ci, run, once, headless or no token says the run finishes.
        self.assertDecisions("gate", tuple(["pnpm", name] for name in (
            "test:ci:watch", "test:watch:ci", "test-watch-run", "test:run:watch", "watch:ci", "test:once:watch",
            "test:watch:headless", "test:no-watch", "test:watch:no-ui",
        )))

    def test_server_prefixes_stay_long_lived_whatever_follows(self):
        # start:ci is typically the server an e2e run talks to.
        self.assertDecisions("direct", tuple(["pnpm", name] for name in (
            "dev", "dev-build", "dev:ci", "dev-ci", "dev:legacy-ci", "start", "start:test", "start:ci", "serve:ci",
            "up-ci", "down-ci",
        )) + (["make", "dev-ci"], ["make", "up-ci"], ["make", "down-ci"]))

    def test_a_check_ahead_of_and_does_not_hide_the_tests_after_it(self):
        self.assertDecisions("gate", tuple(["pnpm", name] for name in (
            "lint-and-test", "format:and:build", "lint:and:test:ci",
        )))
        self.assertDecisions("direct", tuple(["pnpm", name] for name in ("typecheck:tests", "lint:fix", "format:check")))

    def test_vitest_run_after_an_option_value_is_not_a_watcher(self):
        self.assertDecisions("gate", (
            ["vitest", "--config", "vitest.config.ts", "run"],
            ["vitest", "-c", "x", "run"],
            ["vitest", "--root", "web", "run"],
            ["vitest", "--reporter", "verbose", "run"],
            ["vitest", "--no-such-option", "value", "run"],
            ["npx", "--package", "vitest", "vitest", "run"],
            ["npx", "vitest", "--config", "x", "run"],
            ["pnpm", "exec", "vitest", "--root", "web", "run"],
            ["playwright", "--config", "pw.config.ts", "test"],
            # --browser is a flag in vitest, so a word after it is not its value.
            ["vitest", "--browser", "run"],
            ["vitest", "--browser=chromium", "run"],
        ), tty=True)
        # Without a command the same options still start a watcher in a terminal.
        self.assertDecisions("direct", (
            ["vitest", "--config", "vitest.config.ts"],
            ["vitest", "--silent", "src/a.test.ts"],
            ["pnpm", "exec", "vitest", "--root", "web", "src/a.test.ts"],
            ["playwright", "--config", "pw.config.ts", "show-report"],
            ["vitest", "--browser"],
        ), tty=True)
        self.assertDecisions("direct", (["playwright", "--config", "pw.config.ts", "show-report"],))

    def test_version_pinned_runners_are_recognised(self):
        self.assertDecisions("gate", (
            ["npx", "vitest@latest", "run"],
            ["npx", "-y", "playwright@1.50", "test"],
            ["pnpm", "dlx", "vitest@latest", "run"],
            ["npx", "@playwright/test@1.50", "test"],
            ["npx", "@playwright/test", "test"],
        ))
        self.assertDecisions("gate", (["npx", "vitest@latest", "run"], ["pnpm", "dlx", "vitest@2", "--run"]), tty=True)
        self.assertDecisions("direct", (["npx", "vitest@latest"], ["npx", "prettier@3", "--check", "."]), tty=True)
        self.assertDecisions("direct", (["npx", "@playwright/test@1.50", "show-report"], ["npx", "tsc@5", "--noEmit"]))

    def test_npx_and_npm_exec_read_the_command_string_of_the_call_option(self):
        self.assertDecisions("gate", (
            ["npx", "-c", "vitest run"],
            ["npx", "--call", "pytest -q"],
            ["npx", "--call=vitest run"],
            ["npx", "-y", "-c", "playwright test"],
            ["npm", "exec", "-c", "vitest run"],
            ["npm", "exec", "--yes", "--call", "playwright test"],
            ["npx", "-c", "vitest@latest run"],
            ["npx", "--call", "@playwright/test@1.50 test"],
            ["pnpm", "exec", "-c", "vitest run"],
            ["pnpm", "-c", "exec", "vitest run"],
            ["yarn", "exec", "vitest run"],
            # The same watch flags as on the command line: a false value is not a watcher.
            ["npx", "-c", "vitest --watch=false"],
            ["npx", "-c", "jest --watchAll=false"],
            ["npx", "-c", "jest --watchman=false"],
            ["pnpm", "exec", "-c", "vitest run --watchAll=0"],
        ))
        self.assertDecisions("direct", (
            ["npx", "-c", "prettier --check ."],
            ["npm", "exec", "-c", "tsc --noEmit"],
            ["npx", "-c", "vitest --watch"],
            ["npx", "-c", "jest --watchAll"],
            ["pnpm", "exec", "-c", "vitest --ui"],
        ))

    def test_dry_runs_and_questions_run_directly(self):
        self.checkout(self.fixture.root / "nested")
        self.assertDecisions("direct", (
            ["make", "-n", "test"],
            ["make", "--dry-run", "test"],
            ["make", "--just-print", "test"],
            ["make", "--recon", "test"],
            ["make", "-q", "test"],
            ["make", "--question", "test"],
            ["make", "-sn", "test"],
            ["make", "-j", "4", "-n", "test"],
            ["make", "-C", "nested", "-n", "test"],
            ["make", "-kqC", "nested", "test"],
        ))
        # A value that looks like the flag is not the flag.
        self.assertDecisions("gate", (
            ["make", "-s", "test"],
            ["make", "-C", "n", "test"],
            ["make", "-f", "n", "test"],
            ["make", "-I", "q", "test"],
        ))

    def test_poetry_check_only_validates_the_project(self):
        self.assertDecisions("direct", (["poetry", "check"], ["poetry", "-n", "check"]))
        # Yarn Berry's check runs the project's own script, which may include tests.
        self.assertDecisions("gate", (
            ["yarn", "check"],
            ["yarn", "--silent", "check"],
            ["yarn", "run", "check"],
            ["make", "check"],
            ["pnpm", "check"],
            ["npm", "run", "check"],
            ["poetry", "run", "check"],
        ))

    def test_the_gate_runs_the_real_tool_with_its_budget(self):
        result = self.fixture.run(["npm", "test"], explain=False)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(result.stdout.strip(), "npm AGENT_GATE_CPUS=2")
        result = self.fixture.run(["npm", "run", "lint"], explain=False)
        self.assertEqual(result.stdout.strip(), "npm AGENT_GATE_CPUS=none")

    def test_a_missing_gate_refuses_heavy_work(self):
        alone = self.fixture.tmp / "alone"
        alone.mkdir()
        shutil.copy2(BIN / "devshim", alone / "devshim")
        other = Fixture(alone / "devshim")
        self.addCleanup(other.cleanup)
        result = other.run(["npm", "test"], explain=False)
        self.assertEqual(result.returncode, 78)
        self.assertIn("agent-gate is unavailable", result.stderr)
        self.assertEqual(other.run(["npm", "run", "lint"], explain=False).returncode, 7)


class AliasTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()
        self.addCleanup(self.fixture.cleanup)
        self.env = self.fixture.env

    def call(self, *command: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        return subprocess.run(command, env=env or self.env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60)

    def cpus(self) -> Sequence[str]:
        return (PY, "-c", "import os; print(os.environ['AGENT_GATE_CPUS'])")

    def test_agent_verify_keeps_its_contract(self):
        verify = str(BIN / "agent-verify")
        self.assertEqual(self.call(verify, "inherited").returncode, 1)
        inside = self.call(str(BIN / "agent-gate"), "-c", "1", "--", verify, "inherited", "--memory", "2", "--cpus", "1")
        self.assertEqual(inside.returncode, 0, inside.stderr)
        wider = self.call(str(BIN / "agent-gate"), "-c", "1", "--", verify, "inherited", "--cpus", "4")
        self.assertEqual(wider.returncode, 78, wider.stderr)
        for bad in (("--cpus", "nan"), ("--memory", "inf"), ("--timeout", "-1")):
            with self.subTest(bad=bad):
                refused = self.call(verify, "run", "--memory", "1", "--cpus", "1", *bad, "--", "true")
                self.assertEqual(refused.returncode, 78, refused.stderr)
        result = self.call(verify, "--memory", "2", "--cpus", "1.5", "--label", "x", "--", *self.cpus())
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "2"), result.stderr)
        result = self.call(verify, "run", "--memory", "2", "--cpus", "1", "--timeout", "30", "--", *self.cpus())
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "1"), result.stderr)
        refused = self.call(verify, "run", "--cpus", "1", "--resource", "db", "--", "true")
        self.assertEqual(refused.returncode, 78)
        self.assertIn("project-owned", refused.stderr)
        self.assertEqual(self.call(verify, "status", "--json").returncode, 0)

    def test_agent_verify_inherited_judges_the_need_after_the_machine_claim_cap(self):
        # A consumer that asked --cpus 4 and was admitted capped to 2 still finds its own inherited --cpus 4 covered.
        env = dict(self.env, AGENT_GATE_MAX_CLAIM="2")
        verify = str(BIN / "agent-verify")
        gate = str(BIN / "agent-gate")
        nested = self.call(gate, "-c", "4", "--", verify, "inherited", "--cpus", "4", env=env)
        self.assertEqual(nested.returncode, 0, "inherited --cpus N must judge N after the same cap a claim gets: "
                         + nested.stderr)
        small = self.call(gate, "-c", "1", "--", verify, "inherited", "--cpus", "4", env=env)
        self.assertEqual(small.returncode, 78, small.stderr)

    def test_agent_verify_forwards_wait_flags_and_status_arguments_to_agent_gate(self):
        verify = str(BIN / "agent-verify")
        gate = str(BIN / "agent-gate")
        # --wait together with --no-wait is agent-gate's own usage error, whichever tool is asked.
        direct = self.call(gate, "--wait", "5", "--no-wait", "--", "true")
        self.assertEqual(direct.returncode, 2, direct.stderr)
        for order in (("--wait", "5", "--no-wait"), ("--no-wait", "--wait", "5")):
            with self.subTest(order=order):
                both = self.call(verify, "--cpus", "1", *order, "--", "true")
                self.assertEqual(both.returncode, direct.returncode, both.stderr)
        # status hands every argument to agent-gate status, which owns what it accepts.
        listed = self.call(verify, "status", "--json")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(self.call(verify, "status").returncode, 0)
        unknown = self.call(verify, "status", "--json", "--bogus")
        self.assertEqual(unknown.returncode, self.call(gate, "status", "--json", "--bogus").returncode, unknown.stderr)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("--bogus", unknown.stderr)
        # A single wait flag still works.
        waited = self.call(verify, "--cpus", "1", "--wait", "5", "--", *self.cpus())
        self.assertEqual((waited.returncode, waited.stdout.strip()), (0, "1"), waited.stderr)
        unqueued = self.call(verify, "--cpus", "1", "--no-wait", "--", *self.cpus())
        self.assertEqual((unqueued.returncode, unqueued.stdout.strip()), (0, "1"), unqueued.stderr)


if __name__ == "__main__":
    unittest.main()
