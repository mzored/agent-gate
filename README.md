# agent-gate

Several coding agents can start tests and builds at the same time. On a Mac, those jobs can fill the CPU and push memory into swap while you are still using the machine. `agent-gate` queues heavy commands, limits how many run together, and waits when memory is tight. It needs no daemon or Python packages.

macOS and Python 3.11 or newer are supported. Docker is only needed for commands that use `--docker-gib`.

## Run one command

```sh
git clone https://github.com/mzored/agent-gate.git
cd agent-gate
./bin/agent-gate -c 2 -- make test
./bin/agent-gate status
```

The machine has one CPU token per logical CPU. A job reserves 2 tokens by default, so other jobs fit beside it, and its test runner may use every CPU that no other job's runners use: a job alone on an idle Mac runs as fast as it would without the gate. The gate sets `VITEST_MAX_WORKERS` and `PYTEST_XDIST_AUTO_NUM_WORKERS` (for `pytest -n auto`) to that number, and exports it as `AGENT_GATE_CPUS` for other runners. `-c K` reserves K tokens and limits the runners to K workers. One job can reserve at most half the logical CPUs.

While other jobs run, a job that reserves more than one token also waits for free memory. A 1-token job waits only while the Mac is actually swapping or memory pressure is critical. The gate also lowers a running command's CPU priority. Your command keeps its own exit code.

Use `./bin/agent-gate --help` for the other options, including wait limits, a timeout, and Docker VM memory claims. `./bin/agent-gate usage` compares past claims with measured CPU use.

## Route test and build commands through the gate

The optional PATH shims cover `make`, `npm`, `pnpm`, `pytest` and other common tools. They queue tests, builds and similar finite jobs. Installs, linters, development servers and watchers run directly.

The shims also cover Docker and Xcode:

- `docker build`, `docker buildx build` and `bake`, and `docker compose build` queue with a 1.5 GiB claim on the Docker VM (`AGENT_GATE_DOCKER_BUILD_GIB`). `docker compose up --build` runs directly because it keeps running.
- `xcodebuild` builds, tests, archives and analyses queue; `-list`, `-show…` and other queries, and `clean` alone, run directly. Unless the command sets `-jobs`, its build follows the CPUs the gate grants.
- Tests on simulators claim about 1.5 GiB per simulator (`AGENT_GATE_SIMULATOR_GIB`), one per simulator destination. Unless the command sets a test runner count, the shim adds `-maximum-parallel-testing-workers 1`, so parallel testing does not boot a clone per CPU. A command that sets one claims a simulator per runner.
- `xcrun simctl boot` queues with a 1-token claim and one simulator. Other `xcrun` calls run directly.

Every booted simulator holds one token for as long as it runs, whoever booted it, and its memory counts against later claims. `status` lists the booted simulators. Xcode's own window, Simulator.app and tools that call CoreSimulator directly do not pass through the shims.

Add this line to both `~/.zshenv` and `~/.zprofile` so login and non-login shells find the shims first:

```sh
export PATH="$HOME/.local/devshims:$PATH"
```

Then run:

```sh
./bin/gate-host install
./bin/gate-host doctor
DEVSHIM_EXPLAIN=1 npm test
```

`install` creates links in `~/.local/devshims`. It will not replace regular files. `doctor` reports the links and PATH order without changing them. `DEVSHIM_EXPLAIN=1` shows how a command would be routed without running it. Set `DEVSHIM_BYPASS=1` for one command that should skip the shim.

To stop admission temporarily, run `./bin/agent-gate off`. Run `./bin/agent-gate on` to resume it. The switch applies to every gate on the Mac. To remove the shims, run `./bin/gate-host uninstall`; remove the PATH lines yourself. Uninstall leaves running commands and gate state alone.

## Clean up leftovers

`./bin/agent-gate janitor` lists what agents left running that nobody owns, and why each item can go. It changes nothing unless you add `--apply`.

- A process adopted by launchd whose working directory was deleted, or replaced by a new directory at the same path.
- A `log stream` whose output pipe has no reader. One whose filter matches nothing never writes, so it never notices.
- A launchd job of yours, not Apple's, whose working directory is gone.
- A simulator booted longer than `--simulator-hours` (2 by default) with no live owner.

A directory counts as gone only when nothing, or a different directory, is at its path. A directory the janitor cannot read, such as one behind macOS privacy protection or on an unmounted volume, does not count.

The janitor keeps anything that may still have a live owner and says who it is:

- a running parent, session leader or process group leader;
- a live gated run;
- a client connected to a TCP port it serves;
- a Unix socket it listens on.

A simulator is kept while any process above the gated run that booted it still runs, while a process names it, or while Xcode, `xcodebuild`, Simulator or Claude's iOS simulator helper runs.

`--apply` reads the Mac a second time and acts only on what both readings agree on:

- processes get TERM and then KILL, each one's identity checked first;
- jobs are booted out of launchd, and their plist files stay in place;
- simulators are shut down without erasing them.

Each action is logged.

`status` and the janitor also compare Docker Desktop's disk image with the free disk space. They warn when free space falls below `AGENT_GATE_DISK_FLOOR_GIB` (10 GiB by default) or when the image may still grow past the free space. The fix is a lower disk limit in Docker Desktop. Pruning only frees room inside the image. Nothing is pruned automatically.

## Develop

The tests run on macOS with Python's standard library:

```sh
python3 -B -m unittest discover -s tests
```

`bin/agent-verify` keeps the older command name working for existing scripts. New calls should use `agent-gate`.

MIT licensed. See [LICENSE](LICENSE).
