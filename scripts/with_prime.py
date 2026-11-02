"""Run a command using machine-local Prime credentials without printing or copying them into the repo."""

import json
import os
import subprocess
import sys
from pathlib import Path

path = Path.home() / ".local/share/home/circuitdoctor/prime.json"
if not path.exists():
    path = Path.home() / ".prime/config.json"
config = json.loads(path.read_text())
environment = dict(os.environ)
environment["PRIME_API_KEY"] = environment.get("PRIME_API_KEY") or config["api_key"]
if config.get("team_id"):
    environment.setdefault("PRIME_TEAM_ID", config["team_id"])
raise SystemExit(subprocess.run(sys.argv[1:], env=environment).returncode)
