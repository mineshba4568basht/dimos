#!/usr/bin/env python3
from __future__ import annotations

from . import prompt_tools as p
from .installer_status import installer_status
from .shell_tooling import run_command

_already_called_apt_get_update = False


def apt_install(package_names: list[str]) -> None:
    """Install packages with apt-get if needed."""
    global _already_called_apt_get_update
    if not package_names:
        return

    if not _already_called_apt_get_update:
        update_res = run_command(
            ["sudo", "apt-get", "update"],
            print_command=True,
            dry_run=installer_status["dry_run"],
        )
        if update_res.code != 0:
            raise RuntimeError(f"sudo apt-get update failed: {update_res.code}")
        _already_called_apt_get_update = True

    failed_packages: list[str] = []
    for each_pkg in package_names:
        res = run_command(
            ["dpkg", "-s", each_pkg], dry_run=installer_status["dry_run"], capture_output=True
        )
        if res.code == 0:
            if "Status: install ok" in res.stdout:
                p.sub_header(f"- ✅ looks like {p.highlight(each_pkg)} is already installed")
                continue
            else:
                # FIXME: make a list of all invalid apt-get package names
                # p.sub_header(f"- 🟠 looks like {p.highlight(each_pkg)} doesn't")
                continue

        p.sub_header(f"\n- installing {p.highlight(each_pkg)}")
        install_res = run_command(
            ["sudo", "apt-get", "install", "-y", each_pkg],
            print_command=True,
            dry_run=installer_status["dry_run"],
        )
        if install_res.code != 0:
            failed_packages.append(each_pkg)

    if failed_packages:
        cmds = "\n".join(f"    sudo apt-get install -y {pkg}" for pkg in failed_packages)
        raise RuntimeError(
            f"apt-get install failed for: {' '.join(failed_packages)}\n"
            f"Try to install them yourself with\n{cmds}"
        )


__all__ = ["apt_install"]
