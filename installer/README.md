# How can I find+change the thing I care about in the installer?

There are 5 phases, all those are called from `__main__.py`: start there.

The same script (e.g. `installer/__main__.py`) is used for the user-interactive install, the non-interactive install (e.g. the inside of docker), and the template repo setup -- because all of them have tons of overlap.

- phase0: ask the user what features they want (pick native install, docker setup, or nix flake setup)
- phase1: native install (if chosen or if inside of docker)
- phase2: check for vital system dependencies
- phase3: install dimos via `uv pip install dimos[<features>]`
- phase4: dimos sanity checks

The python installer is bootstrapped from a shell based installer which is under `bin/install` (more details on bootstrapping below if that is what you're interested in)

# How do I test my changes to the installer?

- `pip install -r ./installer/requirements.txt` # like 3 things, just enough to build the installer
- run `build_scripts/build_pyz.py` to build the pyz app
- run `python ./installer/installer.pyz` to test it
- to publish your change, you'll have to make a release and you *must* name the pyz file `installer.pyz` and attach it to the release. Also note: github url's can lag. Even when you can view a change on github, the github raw endpoint will take several minutes to update.

## How do I add a pip dependency?

The python app needs to be bundled with all pip dependencies. Just make sure the pip dependency is minimal (no system dependencies) and add it to the `installer/pyz_app/requirements.txt` file. The build command should handle the rest.

# How do I keep the installer up to date?

This is so simple it can (and should) be run in CI.
- FYI: Every pip module has a list of system dependencies (apt, brew, nix)
- The installer reads from the `pyproject.toml` to know which features (ex: `sim`, `cuda`) need which pip modules and (therefore) which system dependencies are needed for the feature as well. Everything (other than the pip-module-to-system-dependency mapping) is derived dynamically from the `pyproject.toml`.
- There is a script `build_scripts/refresh_system_dep_db.py` that will generate the pip-module-to-system-dependency mapping any time a pip module is added anywhere in the `pyproject.toml`. This script using claude to estimate the mapping, then validate the names using cli tools, then gives you the opportunity to fix any mistakes by editing the `installer/pyz_app/bundled_files/pip_dependency_database.json` file (whenever you want).
- Last important thing: 99% of dimos system dependencies are indirect, but there are a few exceptions -- things we depend on directly, no pip module needs them. For those exceptions edit the `installer/pyz_app/support/constants.py` file. Modify lists such as `DEPENDENCY_APT_PACKAGES_SET_MINIMAL` to contain things dimos directly needs. At time of writing (Jan 2026), there are not vars for feature-specific direct dependencies, but follow the usage of `DEPENDENCY_APT_PACKAGES_SET_MINIMAL` (and friends) if you need to add a feature-specific direct dependency.

# How does the very beginning of the installer work? (Bootstrapping)

- The user runs a curl command ex: `sh <(curl -fsSL "https://raw.githubusercontent.com/dimensionalOS/dimos/refs/heads/master/install")`
- This format is very specific because it needs to:
1. preserve STDIN (ex: `curl URL | sh` will not work)
2. be able to accept arguments (ex: `sh <(curl URL) --help` should work)


- The curl command is effectively just running `bin/install`, which is a POSIX shell script (because everything other than windows has a roughly POSIX shell)
- The only job of the `bin/install` is to run the stage 1 (e.g. pyz_app) installer
- To do this it just needs to do three things:
1. Assume absolutely nothing about the system (can't assume `git` exists or `grep`). It's POSIX and only needs the system to have `untar` and either `curl` or `wget`.
2. Download a small python binary (not a full python install) that is a known (exact) version
3. Download the compiled pyz_app (all pip dependencies are bundled) and run it using that python binary

Thats it. The python code handles everything else.
