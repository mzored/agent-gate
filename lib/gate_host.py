"""Install and check the host wiring that agent-gate depends on.

The gate's logic is lib/agent_gate.py. This module declares what the host needs
to reach it and keeps the host in line with that declaration:

- the PATH shims in ~/.local/devshims, linked into this checkout;
- whether admission control is on or off after login (`AT_LOGIN`);
- that a fresh zsh, login or not, finds those shims before the real tools.

    gate-host doctor    report every item; change nothing
    gate-host install   create or repoint the links and the LaunchAgent, then report
                        (refused from any checkout but the live one)
    gate-host uninstall remove only this checkout's links and LaunchAgent

Exit codes: 0 when everything matches, 1 when something does not (after install:
something install may not fix), 2 for a usage error or a refused install.

With AT_LOGIN = "off", install loads the LaunchAgent with RunAtLoad, so `agent-gate off`
runs right away: admission is off host-wide, for every agent, from that moment and not
only from the next login. `agent-gate on` switches it back on.

install never edits shell startup files, never replaces a real file or directory,
and never switches admission on: `agent-gate on` stays the user's command.
uninstall leaves shell startup files, gate state and running commands alone.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import plistlib
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Admission control after login: "off" or "on". The off switch is a file under /tmp,
#: so a reboot clears it; "off" therefore installs a LaunchAgent that runs
#: `agent-gate off` at every login, and loading it also switches admission off at once.
#: "on" removes that LaunchAgent but leaves the current switch alone.
AT_LOGIN = "on"

#: The tools whose entry points the shims take over; each links to `_devshim`.
TOOLS = ("bun", "make", "npm", "npx", "playwright", "pnpm", "poetry", "pytest", "uv", "vitest", "yarn")
SHIM_DIR = Path(".local/devshims")
SHIM_BODY = "_devshim"
LABEL = "dev.agent-gate.off"
#: Stable interpreter paths, in order; a versioned Cellar path would break on upgrade.
PYTHONS = ("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3")
#: What a process starts with before any startup file runs.
BASE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# The live checkout is the one the existing shim links point into (as the main checkout, never
# a linked worktree). A fresh host has no links yet, so it is the main checkout of the
# repository this code runs from. The links, the doctor report and the install guard all use
# that one definition. A link into a checkout that no longer exists is ignored. Test seams: GATE_HOST_ROOT (the live
# checkout, which the links point at), GATE_HOST_AT_LOGIN (overrides AT_LOGIN) and
# GATE_HOST_LAUNCHCTL (the launchctl binary).


def live_root(checkout: Path) -> Path:
    """The checkout to link to: this one, or the main checkout when this is a linked worktree."""
    dotgit = checkout / ".git"
    if dotgit.is_file():
        text = dotgit.read_text().strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            if not gitdir.is_absolute():
                gitdir = (checkout / gitdir).resolve()
            # <main>/.git/worktrees/<name>
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parent.parent.parent
    return checkout


def linked_checkout(shims: Path) -> Optional[Path]:
    """The live checkout the existing shim links point into, if one still exists."""
    for name, target in (("agent-gate", "agent-gate"), (SHIM_BODY, "devshim")):
        link = shims / name
        if not link.is_symlink():
            continue
        path = Path(os.path.normpath(link.parent / os.readlink(link)))
        checkout = path.parent.parent
        # A foreign bin/agent-gate must not become the live checkout: require gate_host too.
        if path.name == target and path.parent.name == "bin" and (checkout / "lib" / "gate_host.py").is_file():
            return live_root(checkout)
    return None


class Host:
    def __init__(self, env: Mapping[str, str]):
        self.env = env
        self.home = Path(env.get("HOME") or Path.home())
        self.checkout = Path(__file__).resolve().parents[1]
        self.shims = self.home / SHIM_DIR
        self.root = Path(env.get("GATE_HOST_ROOT") or linked_checkout(self.shims) or live_root(self.checkout))
        self.at_login = env.get("GATE_HOST_AT_LOGIN") or AT_LOGIN
        self.launchctl = env.get("GATE_HOST_LAUNCHCTL") or "/bin/launchctl"
        self.plist = self.home / "Library/LaunchAgents" / f"{LABEL}.plist"
        self.lines: List[Tuple[str, str]] = []
        self.failed = False

    # ------------------------------------------------------------------ report

    def say(self, status: str, subject: str) -> None:
        self.lines.append((status, subject))
        if status not in ("ok", "info", "fixed"):
            self.failed = True

    def show(self, path: Path) -> str:
        try:
            return "~/" + str(path.relative_to(self.home))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------- links

    def links(self) -> List[Tuple[Path, str]]:
        bin_dir = self.root / "bin"
        declared = [
            (self.shims / SHIM_BODY, str(bin_dir / "devshim")),
            (self.shims / "agent-gate", str(bin_dir / "agent-gate")),
            (self.shims / "agent-verify", str(bin_dir / "agent-verify")),
        ]
        declared += [(self.shims / tool, SHIM_BODY) for tool in TOOLS]
        return declared

    def check_link(self, path: Path, target: str, fix: bool) -> None:
        subject = f"{self.show(path)} -> {target}"
        if path.is_symlink():
            actual = os.readlink(path)
            if actual == target:
                self.say("ok", subject)
                return
            if not fix:
                self.say("wrong", f"{subject} (points at {actual})")
                return
            temporary = path.with_name(f".{path.name}.gate-host")
            if temporary.is_symlink():
                temporary.unlink()
            temporary.symlink_to(target)
            os.replace(temporary, path)
            self.say("fixed", f"{subject} (pointed at {actual})")
        elif path.exists():
            self.say("foreign", f"{subject} (a real {'directory' if path.is_dir() else 'file'} is there; "
                                "move it away, then run install)")
        elif not fix:
            self.say("missing", subject)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
            self.say("fixed", f"{subject} (created)")

    # -------------------------------------------------------------- LaunchAgent

    def python(self) -> str:
        for candidate in PYTHONS:
            if os.access(candidate, os.X_OK):
                return candidate
        return PYTHONS[-1]

    def agent(self) -> Dict[str, object]:
        log = str(self.home / "Library/Logs/agent-gate-off.log")
        return {
            "Label": LABEL,
            "ProcessType": "Background",
            "ProgramArguments": [self.python(), str(self.root / "bin" / "agent-gate"), "off"],
            "RunAtLoad": True,
            "StandardErrorPath": log,
            "StandardOutPath": log,
        }

    def domain(self) -> str:
        return f"gui/{os.getuid()}"

    def launchctl_run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([self.launchctl, *args], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=30)

    def loaded(self) -> bool:
        return self.launchctl_run("print", f"{self.domain()}/{LABEL}").returncode == 0

    def installed_agent(self) -> Optional[object]:
        try:
            with self.plist.open("rb") as handle:
                return plistlib.load(handle)
        except FileNotFoundError:
            return None
        except (plistlib.InvalidFileException, ValueError, OSError):
            return "unreadable"

    def check_agent(self, fix: bool) -> None:
        subject = f"LaunchAgent {LABEL}"
        current = self.installed_agent()
        loaded = self.loaded()
        if self.at_login == "on":
            if current is None and not loaded:
                self.say("ok", f"{subject} absent (admission stays as switched across logins)")
            elif not fix:
                self.say("stale", f"{subject} present (switches admission off at login; install removes it)")
            else:
                if loaded:
                    self.launchctl_run("bootout", f"{self.domain()}/{LABEL}")
                if current is not None:
                    self.plist.unlink()
                self.say("fixed", f"{subject} removed; admission stays as it is now until `agent-gate on`")
            return
        wanted = self.agent()
        if current == wanted and loaded:
            self.say("ok", f"{subject} switches admission off at login")
            return
        problem = ("missing" if current is None else "wrong" if current != wanted else "unloaded")
        if not fix:
            detail = ""
            if problem == "unloaded":
                detail = ", not loaded"
            elif isinstance(current, dict):
                differ = sorted(key for key in set(wanted) | set(current) if wanted.get(key) != current.get(key))
                detail = f", differs in {', '.join(differ)}"
            self.say(problem, f"{subject} ({self.show(self.plist)}{detail})")
            return
        if current != wanted:
            self.plist.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.plist.with_name(f".{self.plist.name}.gate-host")
            temporary.write_bytes(plistlib.dumps(wanted))
            os.replace(temporary, self.plist)
        (self.home / "Library/Logs").mkdir(parents=True, exist_ok=True)
        if loaded:
            self.launchctl_run("bootout", f"{self.domain()}/{LABEL}")
        result = self.launchctl_run("bootstrap", self.domain(), str(self.plist))
        if result.returncode != 0:
            self.say("broken", f"{subject}: launchctl bootstrap exited {result.returncode}: "
                               f"{(result.stderr or result.stdout).strip()}")
            return
        self.say("fixed", f"{subject} {'written and ' if current != wanted else ''}loaded; it runs "
                          "`agent-gate off` now, so admission is off host-wide from this moment and at every login")

    # -------------------------------------------------------------------- PATH

    def check_path(self) -> None:
        """A fresh zsh must reach every declared link before anything else of that name."""
        zsh = "/bin/zsh"
        if not os.access(zsh, os.X_OK):
            self.say("info", "PATH not checked: no /bin/zsh")
            return
        env = {"HOME": str(self.home), "PATH": BASE_PATH}
        for key in ("USER", "LOGNAME", "TMPDIR", "LANG"):
            if key in self.env:
                env[key] = self.env[key]
        for flags in ("-c", "-lc"):
            subject = f"PATH in `zsh {flags}`"
            try:
                result = subprocess.run([zsh, flags, 'print -r -- "$PATH"'], env=env, stdin=subprocess.DEVNULL,
                                        capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                self.say("broken", f"{subject}: zsh did not finish in 30 seconds")
                continue
            lines = result.stdout.strip().splitlines()
            if result.returncode != 0 or not lines:
                self.say("broken", f"{subject}: zsh exited {result.returncode}: {result.stderr.strip()}")
                continue
            directories = lines[-1].split(":")
            behind = []
            for path, _ in self.links():
                found = first_on_path(path.name, directories)
                if found != path:
                    behind.append(f"{path.name} -> {found or 'nothing'}")
            if behind:
                self.say("wrong", f"{subject} reaches {', '.join(behind)}; the startup files must "
                                  f"prepend {self.show(self.shims)}")
            else:
                self.say("ok", f"{subject} reaches the shims first")

    # ------------------------------------------------------------------- state

    def check_switch(self) -> None:
        command = [sys.executable, str(self.root / "bin" / "agent-gate"), "status", "--json"]
        try:
            result = subprocess.run(command, env=dict(self.env), stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, timeout=60)
            off = json.loads(result.stdout)["off"] if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
            result, off = None, None
        if off is None:
            detail = f"exited {result.returncode}: {result.stderr.strip()}" if result else "no answer"
            self.say("broken", f"agent-gate status {detail}")
            return
        self.say("info", f"admission control is {'off' if off else 'on'} now; declared {self.at_login} at login")

    def run(self, fix: bool) -> int:
        if self.at_login not in ("on", "off"):
            print(f"gate-host: GATE_HOST_AT_LOGIN must be on or off, not {self.at_login!r}", file=sys.stderr)
            return 2
        if fix and self.checkout != self.root.resolve():
            print(f"gate-host: install refused. This checkout ({self.checkout}) is not the live one "
                  f"({self.root}), where the host links point. Its declaration would change the host "
                  f"while the links keep running the live code. Merge the change there, then run "
                  f"{self.root / 'bin' / 'gate-host'} install. To move the live checkout on purpose, run "
                  f"install here with GATE_HOST_ROOT={self.checkout}. doctor works from any checkout.",
                  file=sys.stderr)
            return 2
        for path, target in self.links():
            self.check_link(path, target, fix)
        self.check_agent(fix)
        self.check_path()
        self.check_switch()
        for status, subject in self.lines:
            print(f"{status:<8}{subject}")
        return 1 if self.failed else 0

    def uninstall(self) -> int:
        if self.checkout != self.root.resolve():
            print(f"gate-host: uninstall refused. This checkout ({self.checkout}) is not the live one "
                  f"({self.root}).", file=sys.stderr)
            return 2
        body = self.shims / SHIM_BODY
        body_owned = body.is_symlink() and os.readlink(body) == str(self.root / "bin" / "devshim")
        for path, target in self.links():
            if path.is_symlink() and os.readlink(path) == target and (path == body or path.name in
                    ("agent-gate", "agent-verify") or body_owned):
                path.unlink()
                self.say("fixed", f"{self.show(path)} removed")
        current = self.installed_agent()
        arguments = current.get("ProgramArguments") if isinstance(current, dict) else None
        if (isinstance(arguments, list) and len(arguments) == 3 and
                arguments[1:] == [str(self.root / "bin" / "agent-gate"), "off"] and
                current.get("Label") == LABEL):
            if self.loaded():
                result = self.launchctl_run("bootout", f"{self.domain()}/{LABEL}")
                if result.returncode != 0:
                    self.say("broken", f"LaunchAgent {LABEL} could not be unloaded")
            if not self.failed:
                self.plist.unlink()
                self.say("fixed", f"LaunchAgent {LABEL} removed")
        for status, subject in self.lines:
            print(f"{status:<8}{subject}")
        return 1 if self.failed else 0


def first_on_path(name: str, directories: Sequence[str]) -> Optional[Path]:
    for directory in directories:
        if not directory:
            continue
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["doctor"], []):
        return Host(os.environ).run(fix=False)
    if args == ["install"]:
        return Host(os.environ).run(fix=True)
    if args == ["uninstall"]:
        return Host(os.environ).uninstall()
    if args in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    print("usage: gate-host [doctor | install | uninstall]", file=sys.stderr)
    return 2
