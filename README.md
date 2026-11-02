# CircuitDoctor

**Give an agent a failing open model. Inspect its internals, test an intervention, and export a reversible weight patch with the experiment attached.**

The first target is repetitive reasoning in Qwen3.5-0.8B. CircuitDoctor separates published-patch reproduction, open-editor investigation, and final evaluation. Ending a response sooner is not sufficient evidence of better reasoning.

Eight typed operations share one implementation across Python, CLI JSON, and Prime-hosted editors: **observe, capture, compare, inspect, intervene, sweep, export patch, validate**. The investigator receives no arbitrary-code, filesystem, or final-evaluation tool.

## Install and check

Requires Python 3.12–3.13 and `uv`. Use CUDA for experiments; model weights require additional storage. vLLM and the newer-method examples use isolated environments.

```sh
uv sync --extra dev
uv run circuitdoctor doctor
uv run circuitdoctor tools
uv run pytest -q
```

`doctor` checks dependencies without inference charges. Add `--load --config-file configs/qwen35.json` to load the actual target and verify unchanged-generation hook parity.

## Reproduce the published repair

Download the exact base revision and pass its path to the hash-verifying importer:

```sh
uv run python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["*.safetensors","*.index.json"], revision="2fc06364715b967f1860aea9cf38778875588b17"))'
uv run circuitdoctor reproduce /path/from/download/snapshot-directory
uv run circuitdoctor prepare
uv run python -m nltk.downloader -q punkt punkt_tab
uv run circuitdoctor evaluate \
  --data runs/data/evaluation.jsonl --output runs/reproduction/evaluation \
  --patch runs/reproduction/patches/published-p50 --config-file configs/qwen35.json
uv run circuitdoctor report runs/reproduction
```

The bundled edit is attributed CC BY 4.0 research data; the original dump remains untouched. All affected base and rebuilt tensor hashes must match. Of 131,072 touched positions, 130,188 values differ bitwise after BF16 rounding. Reconstructing a patch is distinct from reproducing its benchmark scores.

## Let an open editor investigate

Set `PRIME_API_KEY` in the process environment. Use `PRIME_TEAM_ID` for team credits. Credentials stay outside code, prompts, reports, and Git.

```sh
uv run circuitdoctor investigate \
  --data runs/data/development.jsonl --output runs/editor-20b-seed0 \
  --editor openai/gpt-oss-20b --seed 0 --config-file configs/qwen35.json
uv run circuitdoctor report runs/editor-20b-seed0
```

Re-run the same command to resume saved state. Each run allows 25 diagnostic batches and ten candidate patches; captures and validations consume batches too. Every editor request reserves a conservative cost before dispatch. Unknown request outcomes retain their reservations.

Use `--editor openai/gpt-oss-120b` for the larger editor, `--behavioral-only` to remove internal measurements, or `--method scripted` / `--method random` for controls. Use fresh output directories for changed conditions.

Final evaluation runs separately. Its checkpoints bind to the target revision, patch manifest, example hash, sampling, engine, and sample count. Final results never flow back through investigator tools.

## Apply, inspect, undo

Bundles contain a manifest plus exact original and edited tensors. All hashes are checked before any tensor changes. File commands write new outputs and refuse overwrites; live temporary edits restore originals on exceptions.

```sh
uv run circuitdoctor patch apply original.safetensors runs/editor/patches/PATCH_ID patched.safetensors
uv run circuitdoctor patch revert patched.safetensors runs/editor/patches/PATCH_ID restored.safetensors
uv run circuitdoctor tool observe '{"max_new_tokens":1024}' \
  --data runs/data/development.jsonl --run runs/manual --config-file configs/qwen35.json
uv run circuitdoctor budget
```

`report` produces self-contained HTML that opens without a GPU or server: paired answers, uncertainty, token use, investigation steps, neuron contributions, patch details, and embedded evidence. Missing results are shown as missing.

## Methods, skill, and experiments

- [Method support and isolated examples](docs/methods.md): Jacobian Lens and circuit-tracer on Qwen3-0.6B; unsupported methods are explicitly identified.
- [Experiment protocol](docs/experiments.md) and `configs/experiment.json`: controls, frozen selection, three seeds, budgets, and success criteria.
- [Measured findings](docs/findings.md): actual checks and runs, separate from planned experiments.
- The `circuitdoctor` skill is maintained in Home System. `scripts/export_skill.py` creates a generated portable ZIP; there is one editable skill source.

## Bounded Prime GPU leases

The project ceiling is $500, split across five stages. These are spending limits, not runtime or success guarantees. GPU leases reserve their maximum cost and start an independent deadline/idle watchdog.

```sh
uv run python scripts/prime_lease.py start runs/lease --hours 2
uv run python scripts/remote.py upload runs/lease
uv run python scripts/remote.py run runs/lease 'cd ~/circuitdoctor && uv sync'
uv run python scripts/remote.py download runs/lease
uv run python scripts/prime_lease.py stop runs/lease
```

The helper chooses one H100 under $5/hour. The watchdog terminates after ten minutes without a heartbeat or at the prepaid deadline. Download evidence before stopping. Use one project ledger; reconcile remote inference charges before another campaign. Provider billing takes precedence over wall-clock estimates.

## Sources

[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) supplies the reproduction; [Qwen's model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) documents thinking loops; [Prime documentation](https://docs.primeintellect.ai/inference/overview) defines inference and billing. Related work includes [EvalRx](https://arxiv.org/abs/2610.25767), [Circuit Oracle](https://library.sparai.org/reports/circuit-oracle-automating-attribution-graph-analysis-via-natural-language-queries-uc5yvy/), and [AgenticInterpBench](https://arxiv.org/abs/2606.24026). Tool orchestration alone is not claimed as novel.

Code: MIT. Published patch data: CC BY 4.0. Official IFEval checkers: Apache 2.0. Models and datasets retain their own licenses; see [NOTICE](NOTICE).
