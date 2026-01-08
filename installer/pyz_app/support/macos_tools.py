#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import requests

from . import prompt_tools as p
from .installer_status import installer_status
from .shell_tooling import command_exists, run_command

_already_called_brew_update = False


def ensure_xcode_cli_tools() -> None:
    p.boring_log("- checking Xcode Command Line Tools")
    try:
        run_command(
            ["xcode-select", "-p"], check=True, capture_output=True
        )  # intentionally not part of dry_run
    except Exception:
        if p.ask_yes_no("Install Xcode Command Line Tools now?"):
            res = run_command(
                ["xcode-select", "--install"], check=True, dry_run=installer_status["dry_run"]
            )
            if res.code != 0:
                raise RuntimeError("Failed to trigger Xcode Command Line Tools installation.")


def ensure_homebrew() -> None:
    if command_exists("brew"):
        p.boring_log("- homebrew found")
        return
    ensure_xcode_cli_tools()
    p.boring_log("- homebrew not found")
    if not p.ask_yes_no("Install Homebrew now? (will run the official install script)"):
        raise RuntimeError("Homebrew is required for automatic dependency install.")

    url = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
    dest = Path("/tmp/brew_install.sh")
    p.boring_log(f"- downloading Homebrew installer to {dest}")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download Homebrew installer: HTTP {resp.status_code}")
    dest.write_bytes(resp.content)
    dest.chmod(0o755)

    res = run_command(
        ["/bin/bash", str(dest)],
        check=True,
        print_command=True,
        dry_run=installer_status["dry_run"],
    )
    if res.code != 0:
        raise RuntimeError("Homebrew installation failed.")


def brew_install(package_names: list[str]) -> None:
    global _already_called_brew_update
    if not package_names:
        return

    ensure_homebrew()
    if not _already_called_brew_update:
        p.boring_log("Running brew update")
        res = run_command(
            ["brew", "update"], print_command=True, dry_run=installer_status["dry_run"]
        )
        if res.code != 0:
            raise RuntimeError(f"brew update failed: {res.code}")
        _already_called_brew_update = True

    failed: list[str] = []
    for pkg in package_names:
        res = run_command(["brew", "list", pkg], capture_output=True)  # intentionally not dry_run
        if res.code == 0:
            p.sub_header(f"- ✅ looks like {p.highlight(pkg)} is already installed")
            continue
        p.sub_header(f"\n- installing {p.highlight(pkg)}")
        install_res = run_command(
            ["brew", "install", pkg], print_command=True, dry_run=installer_status["dry_run"]
        )
        if install_res.code != 0:
            failed.append(pkg)

    if failed:
        raise RuntimeError(f"brew install failed for: {' '.join(failed)}")


__all__ = ["brew_install", "ensure_homebrew", "ensure_xcode_cli_tools"]
