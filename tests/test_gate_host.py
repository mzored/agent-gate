"""gate-host against a throwaway home and an isolated gate state."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import gate_host  # noqa: E402

PROFILE = 'export PATH="$HOME/.local/devshims:$PATH"\n'


class Fixture:
    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="gate-host-test-"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        # zshenv for every zsh; zprofile again because /etc/zprofile's path_helper reorders PATH.
        (self.home / ".zshenv").write_text(PROFILE)
        (self.home / ".zprofile").write_text(PROFILE)
        (self.tmp / "memory").write_text("1 80")
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("AGENT_", "DEVSHIM_", "GATE_HOST_"))}
        self.env.update(
            HOME=str(self.home),
            GATE_HOST_ROOT=str(ROOT),
            AGENT_GATE_DIR=str(self.tmp / "state"),
            AGENT_GATE_MEMORY_FILE=str(self.tmp / "memory"),
        )

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT / "bin" / "gate-host"), *args],
                              env=dict(self.env, **extra), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    def gate(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(ROOT / "bin" / "agent-gate"), *args], env=self.env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)

    @property
    def shims(self) -> Path:
        return self.home / ".local/devshims"

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class GateHostTest(unittest.TestCase):
    def setUp(self) -> None:
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)

    def statuses(self, result: subprocess.CompletedProcess) -> dict:
        lines = {}
        for line in result.stdout.splitlines():
            status, _, subject = line.partition(" ")
            lines[subject.strip()] = status
        return lines

    def test_install_wires_an_empty_home_and_doctor_then_passes(self):
        result = self.f.run("install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "_devshim"), str(ROOT / "bin" / "devshim"))
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(ROOT / "bin" / "agent-gate"))
        for tool in gate_host.TOOLS:
            self.assertEqual(os.readlink(self.f.shims / tool), "_devshim")

        doctor = self.f.run("doctor")
        self.assertEqual(doctor.returncode, 0, doctor.stdout + doctor.stderr)
        self.assertNotIn("fixed", doctor.stdout)
        self.assertIn("PATH in `zsh -lc` reaches the shims first", doctor.stdout)

    def test_doctor_reports_an_empty_home_and_changes_nothing(self):
        result = self.f.run()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing ~/.local/devshims/make -> _devshim", result.stdout)
        self.assertFalse((self.f.home / ".local").exists())

    def test_uninstall_removes_only_owned_wiring(self):
        self.assertEqual(self.f.run("install").returncode, 0)
        foreign = self.f.shims / "ubi"
        foreign.write_text("mine\n")
        changed = self.f.shims / "npm"
        changed.unlink()
        changed.symlink_to("/somewhere/else")
        profile = (self.f.home / ".zshenv").read_text()
        state = self.f.tmp / "state"
        state.mkdir()

        result = self.f.run("uninstall")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ("_devshim", "agent-gate", "agent-verify", "make", "pnpm"):
            self.assertFalse((self.f.shims / name).is_symlink(), name)
        self.assertEqual(os.readlink(changed), "/somewhere/else")
        self.assertEqual(foreign.read_text(), "mine\n")
        self.assertEqual((self.f.home / ".zshenv").read_text(), profile)
        self.assertTrue(state.exists())
        self.assertEqual(self.f.run("uninstall").returncode, 0)

    def test_install_leaves_a_real_file_in_place(self):
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "make").write_text("mine\n")
        result = self.f.run("install")
        self.assertEqual(result.returncode, 1)
        self.assertIn("foreign ~/.local/devshims/make", result.stdout)
        self.assertEqual((self.f.shims / "make").read_text(), "mine\n")
        self.assertEqual(os.readlink(self.f.shims / "npm"), "_devshim")

    def test_install_repoints_a_wrong_link_and_leaves_other_tools_alone(self):
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "agent-gate").symlink_to("/nowhere/agent-gate")
        (self.f.shims / "cargo").symlink_to("_devshim")
        (self.f.shims / "ubi").write_text("#!/bin/sh\n")
        doctor = self.f.run("doctor")
        self.assertIn("wrong", self.statuses(doctor).get(
            f"~/.local/devshims/agent-gate -> {ROOT / 'bin' / 'agent-gate'} (points at /nowhere/agent-gate)"))
        self.assertNotIn("cargo", doctor.stdout)

        result = self.f.run("install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(ROOT / "bin" / "agent-gate"))
        self.assertEqual(os.readlink(self.f.shims / "cargo"), "_devshim")
        self.assertTrue((self.f.shims / "ubi").exists(), "a file that is not a shim is not ours")

    def test_doctor_flags_shims_behind_the_system_path_in_a_login_shell(self):
        self.assertEqual(self.f.run("install").returncode, 0)
        (self.f.home / ".zprofile").write_text("")
        result = self.f.run("doctor")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ok      PATH in `zsh -c` reaches the shims first", result.stdout)
        self.assertRegex(result.stdout, r"wrong   PATH in `zsh -lc` reaches .*make -> /usr/bin/make")

    def test_an_unknown_command_is_a_usage_error(self):
        self.assertEqual(self.f.run("frobnicate").returncode, 2)


class LiveCheckoutTest(unittest.TestCase):
    """install changes the host for every agent, so only the live checkout may run it."""

    def setUp(self) -> None:
        self.f = Fixture()
        self.addCleanup(self.f.cleanup)
        self.main = self.f.tmp.resolve() / "main"
        self.worktree = self.main / ".worktrees/topic"
        for checkout in (self.main, self.worktree):
            for name in ("bin", "lib"):
                shutil.copytree(ROOT / name, checkout / name, ignore=shutil.ignore_patterns("__pycache__"))
        (self.main / ".git/worktrees/topic").mkdir(parents=True)
        (self.worktree / ".git").write_text(f"gitdir: {self.main / '.git/worktrees/topic'}\n")

    def run_from(self, checkout: Path, *args: str, **extra: str) -> subprocess.CompletedProcess:
        env = {key: value for key, value in self.f.env.items() if key != "GATE_HOST_ROOT"}
        return subprocess.run([sys.executable, str(checkout / "bin" / "gate-host"), *args],
                              env=dict(env, **extra), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=120)

    def assert_host_untouched(self) -> None:
        self.assertFalse((self.f.home / ".local").exists())

    def test_install_from_a_linked_worktree_is_refused_and_names_the_live_checkout(self):
        result = self.run_from(self.worktree, "install")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(str(self.main), result.stderr)
        self.assertEqual(result.stdout, "")
        self.assert_host_untouched()

    def test_install_from_a_checkout_that_is_not_the_declared_root_is_refused(self):
        elsewhere = self.f.tmp / "elsewhere"
        result = self.f.run("install", GATE_HOST_ROOT=str(elsewhere))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(str(elsewhere), result.stderr)
        self.assert_host_untouched()

    def test_install_from_the_main_checkout_wires_a_fresh_host(self):
        result = self.run_from(self.main, "install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(self.main / "bin" / "agent-gate"))

    def test_install_from_a_second_clone_is_refused_when_the_links_point_elsewhere(self):
        clone = self.f.tmp.resolve() / "clone"
        for name in ("bin", "lib"):
            shutil.copytree(ROOT / name, clone / name, ignore=shutil.ignore_patterns("__pycache__"))
        self.f.shims.mkdir(parents=True)
        for link, target in (("agent-gate", self.main / "bin" / "agent-gate"),
                             ("_devshim", self.main / "bin" / "devshim")):
            (self.f.shims / link).symlink_to(target)
            with self.subTest(link=link):
                result = self.run_from(clone, "install")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(str(self.main), result.stderr)
                self.assertEqual(os.readlink(self.f.shims / link), str(target))
                self.assertEqual([path.name for path in self.f.shims.iterdir()], [link])
            (self.f.shims / link).unlink()

    def test_doctor_from_a_second_clone_judges_the_host_against_the_live_links(self):
        clone = self.f.tmp.resolve() / "clone"
        for name in ("bin", "lib"):
            shutil.copytree(ROOT / name, clone / name, ignore=shutil.ignore_patterns("__pycache__"))
        self.assertEqual(self.run_from(self.main, "install").returncode, 0)
        doctor = self.run_from(clone, "doctor")
        self.assertEqual(doctor.returncode, 0, doctor.stdout + doctor.stderr)

    def test_links_into_a_checkout_that_is_gone_do_not_block_a_new_clone(self):
        clone = self.f.tmp.resolve() / "clone"
        for name in ("bin", "lib"):
            shutil.copytree(ROOT / name, clone / name, ignore=shutil.ignore_patterns("__pycache__"))
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "agent-gate").symlink_to(self.f.tmp / "removed" / "bin" / "agent-gate")
        result = self.run_from(clone, "install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(clone / "bin" / "agent-gate"))

    def make_clone(self) -> Path:
        clone = self.f.tmp.resolve() / "clone"
        for name in ("bin", "lib"):
            shutil.copytree(ROOT / name, clone / name, ignore=shutil.ignore_patterns("__pycache__"))
        return clone

    def test_the_refusal_names_the_way_to_move_the_live_checkout(self):
        result = self.run_from(self.worktree, "install")
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"GATE_HOST_ROOT={self.worktree}", result.stderr)

    def test_links_into_a_linked_worktree_make_its_main_checkout_the_live_one(self):
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "agent-gate").symlink_to(self.worktree / "bin" / "agent-gate")
        refused = self.run_from(self.worktree, "install")
        self.assertEqual(refused.returncode, 2, refused.stdout + refused.stderr)
        self.assertIn(str(self.main), refused.stderr)
        result = self.run_from(self.main, "install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(self.main / "bin" / "agent-gate"))

    def test_a_vanished_agent_gate_link_leaves_the_devshim_link_to_decide(self):
        clone = self.make_clone()
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "agent-gate").symlink_to(self.f.tmp / "removed" / "bin" / "agent-gate")
        (self.f.shims / "_devshim").symlink_to(self.main / "bin" / "devshim")
        result = self.run_from(clone, "install")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(str(self.main), result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "_devshim"), str(self.main / "bin" / "devshim"))

    def test_a_foreign_directory_with_only_bin_agent_gate_is_never_the_live_checkout(self):
        foreign = self.f.tmp.resolve() / "foreign"
        (foreign / "bin").mkdir(parents=True)
        (foreign / "bin" / "agent-gate").write_text("#!/bin/sh\n")
        self.f.shims.mkdir(parents=True)
        (self.f.shims / "agent-gate").symlink_to(foreign / "bin" / "agent-gate")
        result = self.run_from(self.main, "install")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(os.readlink(self.f.shims / "agent-gate"), str(self.main / "bin" / "agent-gate"))

    def test_doctor_runs_from_a_linked_worktree_and_changes_nothing(self):
        result = self.run_from(self.worktree, "doctor")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(f"missing ~/.local/devshims/agent-gate -> {self.main / 'bin' / 'agent-gate'}", result.stdout)
        self.assert_host_untouched()


class LiveRootTest(unittest.TestCase):
    def test_a_linked_worktree_wires_the_host_to_its_main_checkout(self):
        tmp = Path(tempfile.mkdtemp(prefix="gate-host-root-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        main = tmp / "main"
        (main / ".git/worktrees/topic").mkdir(parents=True)
        worktree = main / ".worktrees/topic"
        worktree.mkdir(parents=True)
        (worktree / ".git").write_text(f"gitdir: {main / '.git/worktrees/topic'}\n")
        self.assertEqual(gate_host.live_root(worktree), main)
        self.assertEqual(gate_host.live_root(main), main)


if __name__ == "__main__":
    unittest.main()
