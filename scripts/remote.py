"""Upload and run CircuitDoctor over SSH while keeping the bounded lease heartbeat alive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from circuitdoctor.storage import read_json

REPO = Path(__file__).resolve().parents[1]


def connection(lease_root: Path):
    lease = read_json(lease_root / "lease.json")
    if lease.get("terminated") or time.time() >= lease["deadline"]:
        raise RuntimeError("Lease is terminated or expired")
    ssh = lease.get("verified_ssh")
    if not ssh:
        status = json.loads(
            subprocess.check_output(
                ["prime", "--plain", "pods", "status", lease["pod_id"], "--output", "json"], text=True
            )
        )
        if status["name"] != lease["name"] or status["ssh"] in (None, "N/A"):
            raise RuntimeError("Pod not ready for SSH")
        ssh = status["ssh"]
        from circuitdoctor.storage import write_json

        lease["verified_ssh"] = ssh
        write_json(lease_root / "lease.json", lease)
    words = shlex.split(ssh)
    if words and words[0] == "ssh":
        words = words[1:]
    # Only accept a simple provider-provided user@address and optional port.
    target = next((w for w in words if "@" in w and not w.startswith("-")), None)
    if target is None:
        raise ValueError("Unrecognized SSH connection")
    port = words[words.index("-p") + 1] if "-p" in words else "22"
    cfg = read_json(Path.home() / ".prime/config.json")
    key = Path(cfg.get("ssh_key_path", ""))
    if not key.is_file():
        key = Path.home() / ".ssh/id_ed25519"
    known_hosts = REPO / "runs/known_hosts"
    control = Path.home() / ".local/share/home/circuitdoctor"
    control.mkdir(parents=True, exist_ok=True, mode=0o700)
    return [
        "ssh",
        "-i",
        str(key),
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=600",
        "-o",
        f"ControlPath={control}/h{hashlib.sha256((target + port).encode()).hexdigest()[:12]}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        target,
    ]


def run(lease_root, command, *, stdin=None, output=None):
    (lease_root / "heartbeat").touch()
    process = subprocess.Popen(
        [*connection(lease_root), command],
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=output,
    )
    if stdin is not None:
        process.stdin.write(stdin)
        process.stdin.close()
    while process.poll() is None:
        (lease_root / "heartbeat").touch()
        time.sleep(5)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def upload(lease_root):
    archive = REPO / "artifacts/source.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name in (
            "src",
            "configs",
            "scripts",
            "examples",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "tests",
        ):
            tar.add(
                REPO / name,
                arcname=name,
                filter=lambda info: (
                    None if ("__pycache__" in info.name or "/backlog/" in info.name) else info
                ),
            )
    run(lease_root, "mkdir -p ~/circuitdoctor && tar -xzf - -C ~/circuitdoctor", stdin=archive.read_bytes())
    dev = REPO / "runs/data"
    if dev.exists():
        with tarfile.open(REPO / "artifacts/data.tar.gz", "w:gz") as tar:
            tar.add(dev, arcname="data")
        run(
            lease_root,
            "mkdir -p ~/circuitdoctor/runs && tar -xzf - -C ~/circuitdoctor/runs",
            stdin=(REPO / "artifacts/data.tar.gz").read_bytes(),
        )


def credentials(lease_root):
    cfg = read_json(Path.home() / ".prime/config.json")
    private = {"api_key": cfg["api_key"], "team_id": cfg.get("team_id")}
    run(
        lease_root,
        "umask 077; mkdir -p ~/.local/share/home/circuitdoctor; "
        "cat > ~/.local/share/home/circuitdoctor/prime.json",
        stdin=json.dumps(private).encode(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "run", "credentials", "download"])
    parser.add_argument("lease", type=Path)
    parser.add_argument("command", nargs="?", default="true")
    args = parser.parse_args()
    if args.action == "upload":
        upload(args.lease)
    elif args.action == "credentials":
        credentials(args.lease)
    elif args.action == "download":
        with (REPO / "artifacts/remote-runs.tar.gz").open("wb") as f:
            run(args.lease, "cd ~/circuitdoctor && tar -czf - runs", output=f)
    else:
        run(args.lease, args.command)
