"""Bound a Prime GPU lease by a durable cost reservation, idle timeout, and independent watchdog.

Never prints credentials. Only pods whose ID and task-specific name match the lease can be terminated.
Run with the project's Python. The watchdog uses the already authenticated Prime CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from circuitdoctor.budget import Ledger
from circuitdoctor.storage import read_json, write_json


def prime(*args):
    result = subprocess.run(
        ["prime", "--plain", *args], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(result.stdout)


def terminate(root):
    lease = read_json(root / "lease.json")
    if lease.get("terminated"):
        return lease
    pods = prime("pods", "list", "--output", "json").get("pods", [])
    match = next((p for p in pods if p["id"] == lease["pod_id"]), None)
    if match is not None:
        if match["name"] != lease["name"]:
            raise RuntimeError("Lease ownership check failed")
        subprocess.run(
            ["prime", "--plain", "pods", "terminate", lease["pod_id"], "--yes"], check=True, timeout=120
        )
    lease.update(terminated=True, ended_at=time.time())
    cost = (lease["ended_at"] - lease["started_at"]) * lease["hourly_usd"] / 3600
    lease.update(
        estimated_cost_usd=cost, cost_source="wall clock x quoted hourly rate; reconcile provider billing"
    )
    write_json(root / "lease.json", lease)
    reservations = lease.get("reservations", [{"id": lease["reservation"], "usd": lease["reserved_usd"]}])
    remaining = cost
    ledger = Ledger(Path(lease["ledger"]))
    for index, reservation in enumerate(reservations):
        charged = remaining if index == len(reservations) - 1 else min(remaining, reservation["usd"])
        ledger.settle(reservation["id"], max(0, charged))
        remaining -= charged
    if cost > lease["reserved_usd"]:
        raise RuntimeError("Lease ran beyond reservation; manual billing reconciliation required")
    return lease


def start(root, ledger_path, stage, hours):
    if root.exists():
        raise FileExistsError("Use a fresh lease directory")
    root.mkdir(parents=True)
    # Avoid provisioning an inaccessible machine. Never replace another primary key.
    cfg = read_json(Path.home() / ".prime/config.json")
    private = Path(cfg.get("ssh_key_path", ""))
    if not private.is_file():
        private = Path.home() / ".ssh/id_ed25519"
    public = Path(str(private) + ".pub")
    if not private.is_file() or not public.is_file():
        raise RuntimeError("No local SSH keypair; configure an existing key before provisioning")
    endpoint = cfg.get("base_url", "https://api.primeintellect.ai").rstrip("/") + "/api/v1/ssh_keys/"
    headers = {"Authorization": "Bearer " + cfg["api_key"]}
    response = httpx.get(endpoint, headers=headers, timeout=30)
    response.raise_for_status()
    keys = response.json().get("data", [])
    local = public.read_text().split()[:2]
    if not keys:
        response = httpx.post(
            endpoint,
            headers=headers,
            json={"name": "CircuitDoctor local public key", "publicKey": public.read_text().strip()},
            timeout=30,
        )
        response.raise_for_status()
        keys = [response.json()]
    if not any(k.get("isPrimary") and k.get("publicKey", "").split()[:2] == local for k in keys):
        raise RuntimeError("Prime primary SSH key differs from this machine; no primary key was changed")
    offers = prime("availability", "list", "--gpu-type", "H100_80GB", "--gpu-count", "1", "--output", "json")[
        "gpu_resources"
    ]
    offers = [o for o in offers if o["stock_status"] == "Available" and o["price_value"] <= 5]
    if not offers:
        raise RuntimeError("No H100 offer under $5/hour")
    offer = min(offers, key=lambda o: o["price_value"])
    # Include a two-minute termination margin inside the reservation.
    reserved = offer["price_value"] * hours
    reservation = Ledger(ledger_path).reserve(stage, "single-H100 lease", reserved)
    name = "circuitdoctor-" + str(int(time.time()))
    started = time.time()
    write_json(root / "offer.json", offer)
    cmd = [
        "prime",
        "--plain",
        "pods",
        "create",
        "--id",
        offer["id"],
        "--name",
        name,
        "--image",
        "ubuntu_22_cuda_12",
        "--disk-size",
        "200",
        "--vcpus",
        "26",
        "--memory",
        "200",
        "--yes",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    (root / "create.log").write_text(result.stdout + result.stderr)
    pods = prime("pods", "list", "--output", "json")["pods"]
    matches = [p for p in pods if p["name"] == name]
    if len(matches) != 1:
        if result.returncode != 0:
            Ledger(ledger_path).settle(reservation, 0)
        raise RuntimeError("Could not identify created pod; inspect create.log before retrying")
    lease = {
        "name": name,
        "pod_id": matches[0]["id"],
        "started_at": started,
        "deadline": started + hours * 3600 - 120,
        "hourly_usd": offer["price_value"],
        "reserved_usd": reserved,
        "reservation": reservation,
        "ledger": str(ledger_path.resolve()),
        "stage": stage,
        "terminated": False,
    }
    write_json(root / "lease.json", lease)
    (root / "heartbeat").touch()
    log = (root / "watchdog.log").open("a")
    process = subprocess.Popen(
        [sys.executable, __file__, "watch", str(root.resolve())],
        stdout=log,
        stderr=log,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    lease["watchdog_pid"] = process.pid
    write_json(root / "lease.json", lease)
    return lease


def extend(root, stage, hours):
    lease = read_json(root / "lease.json")
    if lease.get("terminated") or time.time() >= lease["deadline"] or hours <= 0:
        raise ValueError("Only an active lease can be extended by a positive duration")
    amount = lease["hourly_usd"] * hours
    key = Ledger(Path(lease["ledger"])).reserve(stage, "GPU lease extension", amount)
    lease.setdefault("reservations", [{"id": lease["reservation"], "usd": lease["reserved_usd"]}])
    lease["reservations"].append({"id": key, "usd": amount})
    lease["reserved_usd"] += amount
    lease["deadline"] += hours * 3600
    write_json(root / "lease.json", lease)
    (root / "heartbeat").touch()
    old_pid = lease.get("watchdog_pid")
    log = (root / "watchdog.log").open("a")
    process = subprocess.Popen(
        [sys.executable, __file__, "watch", str(root.resolve())],
        stdout=log,
        stderr=log,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
    )
    lease["watchdog_pid"] = process.pid
    write_json(root / "lease.json", lease)
    if old_pid:
        command = subprocess.run(
            ["ps", "-p", str(old_pid), "-o", "command="], capture_output=True, text=True
        ).stdout
        if "prime_lease.py watch" in command and str(root.resolve()) in command:
            try:
                os.kill(old_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return lease


def watch(root):
    while True:
        lease = read_json(root / "lease.json")
        if lease.get("terminated"):
            return
        idle = time.time() - (root / "heartbeat").stat().st_mtime
        if time.time() >= lease["deadline"] or idle > 600:
            try:
                terminate(root)
                return
            except Exception as exc:
                print(type(exc).__name__, str(exc), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "watch", "stop", "heartbeat", "extend"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--ledger", type=Path, default=Path("runs/project.sqlite"))
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if args.action == "start":
        print(json.dumps(start(args.root, args.ledger, args.stage, args.hours), indent=2))
    elif args.action == "extend":
        print(json.dumps(extend(args.root, args.stage, args.hours), indent=2))
    elif args.action == "stop":
        print(json.dumps(terminate(args.root), indent=2))
    elif args.action == "heartbeat":
        (args.root / "heartbeat").touch()
    else:
        watch(args.root)
