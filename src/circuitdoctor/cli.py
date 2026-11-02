from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .budget import Ledger
from .schema import RunConfig
from .storage import read_examples, read_json, write_json

app = typer.Typer(
    no_args_is_help=True, help="Investigate model failures and export tested, reversible weight repairs."
)
patch_app = typer.Typer(no_args_is_help=True)
app.add_typer(patch_app, name="patch")


def emit(value):
    typer.echo(json.dumps(value, indent=2, default=str))


def config(path: Path | None):
    return RunConfig.model_validate(read_json(path)) if path else RunConfig()


@app.command()
def doctor(config_file: Path | None = None, load: bool = False, output: Path = Path("runs/doctor.json")):
    """Check dependencies; --load verifies real model loading and unchanged hook parity."""
    import importlib.metadata

    import torch

    result = {
        "version": __version__,
        "python": platform.python_version(),
        "cuda": torch.cuda.is_available(),
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "safetensors", "math-verify")
        },
    }
    if load:
        from .runtime import Runtime
        from .schema import Example

        runtime = Runtime(config(config_file))
        result.update(
            target=runtime.describe(), parity=runtime.parity([Example(id="parity", prompt="What is 2 + 3?")])
        )
    write_json(output, result)
    emit(result)


@app.command()
def prepare(output: Path = Path("runs/data"), seed: int = 0, development_n: int = 32, full: bool = False):
    """Create frozen development/validation/test manifests from official datasets."""
    from .data import prepare as run

    manifest = run(output, seed, development_n, full=full)
    emit({"manifest": str(output / "manifest.json"), "counts": manifest["counts"], "seed": seed})


@app.command()
def reproduce(
    base_file: Annotated[Path | None, typer.Argument()] = None,
    output: Path = Path("runs/reproduction/patches/published-p50"),
    source: Path | None = None,
):
    """Rebuild the supplied published p50 patch, verifying all base and edited tensor hashes."""
    from .reproduce import reproduce as run

    emit(run(base_file, output, source).model_dump())


@app.command()
def investigate(
    data: Path,
    output: Path,
    config_file: Path | None = None,
    editor: str = "openai/gpt-oss-20b",
    method: str = "agent",
    seed: int = 0,
    ledger: Path = Path("runs/project.sqlite"),
    behavioral_only: bool = False,
):
    """Run or resume an investigation; data must contain development and validation examples only."""
    from .runtime import Runtime
    from .tools import ToolRegistry

    cfg = config(config_file)
    cfg.sampling.seed = seed
    registry = ToolRegistry(
        Runtime(cfg), read_examples(data), output, Ledger(ledger), behavioral_only=behavioral_only
    )
    write_json(output / "config.json", cfg.model_dump())
    if method == "agent":
        from .agent import investigate as run
        from .prime import PrimeClient

        result = run(registry, PrimeClient(registry.ledger, "experiments", model=editor))
    elif method in ("scripted", "random"):
        from .baselines import scripted

        result = scripted(registry, random_search=method == "random", seed=seed)
        write_json(output / "selection.json", result)
    else:
        raise typer.BadParameter("method must be agent, scripted, or random")
    from .selection import select_patch

    result = {**select_patch(registry), "editor_result": result}
    write_json(output / "selection.json", result)
    write_json(output / "spend.json", registry.ledger.summary())
    emit(result)


@app.command()
def evaluate(
    data: Path,
    output: Path,
    patch: Path | None = None,
    config_file: Path | None = None,
    samples: int = 1,
    engine: str = "transformers",
):
    """Evaluate a frozen patch on test examples; checkpoints bind to the immutable protocol."""
    from .evaluate import evaluate as run

    if engine == "vllm":
        from .vllm_runtime import VLLMRuntime as Backend
    elif engine == "transformers":
        from .runtime import Runtime as Backend
    else:
        raise typer.BadParameter("engine must be transformers or vllm")
    emit(run(Backend(config(config_file)), read_examples(data), output, patch=patch, samples=samples))


@app.command()
def report(run: Path, output: Path | None = None):
    """Render an offline interactive replay from actual run evidence."""
    from .report import report as render

    emit(render(run, output or run / "report.html"))


@app.command()
def tools(config_file: Path | None = None):
    """Print JSON tool schemas without loading model weights."""
    from .tools import ToolRegistry

    emit(
        [
            {
                "name": name,
                "description": ToolRegistry.descriptions[name],
                "parameters": cls.model_json_schema(),
            }
            for name, cls in ToolRegistry.types.items()
        ]
    )


@app.command("tool")
def execute_tool(
    name: str,
    arguments: str,
    data: Path,
    run: Path,
    config_file: Path | None = None,
    ledger: Path = Path("runs/project.sqlite"),
):
    """Execute one typed investigator tool with JSON arguments; resume run-local artifacts."""
    from .runtime import Runtime
    from .tools import ToolRegistry

    registry = ToolRegistry(Runtime(config(config_file)), read_examples(data), run, Ledger(ledger))
    emit(registry.execute(name, json.loads(arguments)).model_dump())


@app.command()
def budget(ledger: Path = Path("runs/project.sqlite")):
    emit(Ledger(ledger).summary())


def patch_file(input: Path, bundle: Path, output: Path, revert: bool):
    from safetensors.torch import load_file, save_file

    from .patch import apply_patch

    if output.exists():
        raise typer.BadParameter("Output exists; refusing overwrite")
    tensors = load_file(input)
    manifest = apply_patch(tensors, bundle, revert=revert)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output, metadata={"format": "pt"})
    emit({"output": str(output), "reverted": revert, "changed_entries": manifest.changed_entries})


@patch_app.command("apply")
def patch_apply(input: Path, bundle: Path, output: Path):
    """Write a patched safetensors file; the input remains untouched."""
    patch_file(input, bundle, output, False)


@patch_app.command("revert")
def patch_revert(input: Path, bundle: Path, output: Path):
    """Verify edited tensor hashes and write the exact original tensors."""
    patch_file(input, bundle, output, True)


if __name__ == "__main__":
    app()
