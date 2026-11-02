from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, select_autoescape

from .storage import read_json


def artifact_files(root: Path):
    # Prune before descent; never inventory backlog trees or symlink targets.
    for directory, dirs, files_in_directory in os.walk(root, followlinks=False):
        dirs[:] = [
            name
            for name in dirs
            if name != "backlog"
            and not name.startswith(".")
            and "backlog" not in (Path(directory) / name).resolve().parts
        ]
        for name in files_in_directory:
            path = Path(directory) / name
            if "backlog" not in path.resolve().parts:
                yield path


def report(root: Path, output: Path):
    data = {
        "title": root.name,
        "events": [],
        "evaluations": [],
        "patches": [],
        "observations": [],
        "editor": None,
        "answer_pairs": [],
    }
    artifacts = sorted(artifact_files(root))
    for eventfile in (p for p in artifacts if p.name == "events.jsonl"):
        condition = str(eventfile.parent.relative_to(root))
        data["events"].extend(
            {**json.loads(line), "condition": condition, "source": str(eventfile.relative_to(root))}
            for line in eventfile.read_text().splitlines()
            if line.strip()
        )
    data["events"].sort(key=lambda event: event.get("time", 0))
    for path in (p for p in artifacts if p.name == "results.json"):
        data["evaluations"].append({"file": str(path.relative_to(root)), "data": read_json(path)})
    for path in (p for p in artifacts if p.name == "manifest.json" and p.parent.parent.name == "patches"):
        data["patches"].append({"file": str(path.relative_to(root)), "data": read_json(path)})
    observation_files = [p for p in artifacts if p.parent.name == "observations" and p.suffix == ".json"]
    data["observations_total"] = len(observation_files)
    for path in observation_files[:20]:
        data["observations"].append({"file": str(path.relative_to(root)), "data": read_json(path)})
    data["conditions"] = [
        {"file": str(p.relative_to(root)), "data": read_json(p)}
        for p in artifacts
        if p.name == "selection.json"
    ]
    if (root / "editor.json").exists():
        data["editor"] = read_json(root / "editor.json")
    for name in ("spend", "selection", "findings"):
        if (root / f"{name}.json").exists():
            data[name] = read_json(root / f"{name}.json")
    for baseline in (p for p in artifacts if p.parent.name == "baseline" and p.suffix == ".json"):
        edited = baseline.parent.parent / "edited" / baseline.name
        if not edited.exists():
            continue
        a, b = read_json(baseline), read_json(edited)
        by_key = {(r["id"], r.get("sample", 0)): r for r in b["rows"]}
        for row in a["rows"]:
            other = by_key.get((row["id"], row.get("sample", 0)))
            if other is not None:
                data["answer_pairs"].append(
                    {
                        "label": row["id"] + " · " + str(baseline.parent.parent.relative_to(root)),
                        "original": row["text"],
                        "edited": other["text"],
                        "evidence": {
                            "baseline_file": str(baseline.relative_to(root)),
                            "edited_file": str(edited.relative_to(root)),
                            "baseline": row,
                            "edited": other,
                        },
                    }
                )

    # Bound the offline viewer while retaining every full trace in the evidence directory.
    def excerpt(text):
        return (
            text
            if len(text) <= 16000
            else text[:8000] + "\n[Excerpt: full trace in evidence file]\n" + text[-8000:]
        )

    pairs = data["answer_pairs"]
    pairs.sort(
        key=lambda pair: (
            not (pair["evidence"]["baseline"].get("correct") != pair["evidence"]["edited"].get("correct")),
            pair["label"],
        )
    )
    data["answer_pairs_total"] = len(pairs)
    data["answer_pairs"] = pairs[:50]
    for pair in data["answer_pairs"]:
        pair["excerpted"] = len(pair["original"]) > 16000 or len(pair["edited"]) > 16000
        pair["original"], pair["edited"] = excerpt(pair["original"]), excerpt(pair["edited"])
        pair["evidence"]["baseline"] = {k: v for k, v in pair["evidence"]["baseline"].items() if k != "text"}
        pair["evidence"]["edited"] = {k: v for k, v in pair["evidence"]["edited"].items() if k != "text"}
    for observation in data["observations"]:
        for row in observation["data"]["rows"]:
            row["text"] = excerpt(row["text"])
    # JSON evidence is embedded: the report remains navigable offline after export.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (files("circuitdoctor") / "templates/report.html").read_text()
    env = Environment(autoescape=select_autoescape(["html"]))
    html = env.from_string(template).render(title=data["title"], payload=payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return {"report": str(output), "events": len(data["events"]), "evaluations": len(data["evaluations"])}
