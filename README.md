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

`-c` is the number of CPU tokens the job asks for. The default is 2. One job can claim at most half the machine's logical CPUs, so another job can run beside it. The gate also lowers a running command's CPU priority. Your command keeps its own exit code.

Use `./bin/agent-gate --help` for the other options, including wait limits, a timeout, and Docker VM memory claims. `./bin/agent-gate usage` compares past claims with measured CPU use.

## Route test and build commands through the gate

The optional PATH shims cover `make`, `npm`, `pnpm`, `pytest` and other common tools. They queue tests, builds and similar finite jobs. Installs, linters, development servers and watchers run directly.

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

To stop admission temporarily, run `./bin/agent-gate off`. Run `./bin/agent-gate on` to resume it. The switch applies to every gate on the Mac. To remove the shims and this project's login agent, run `./bin/gate-host uninstall`; remove the PATH lines yourself. Uninstall leaves running commands and gate state alone.

## Develop

The tests run on macOS with Python's standard library:

```sh
python3 -B -m unittest discover -s tests
```

`bin/agent-verify` keeps the older command name working for existing scripts. New calls should use `agent-gate`.

MIT licensed. See [LICENSE](LICENSE).
