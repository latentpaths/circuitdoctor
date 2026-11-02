# Method support

Implementation status is separate from actual integration verification; measured runs are recorded in findings.

| Method | Implementation | First target / limitation |
|---|---|---|
| Behavior, loop metrics, grading | Core | Qwen3.5-0.8B |
| Residual/MLP/neuron capture | Native hooks | Shape and unchanged-generation checks required |
| Mean contrast / PCA | Core | Association; causal tests required |
| Logit lens / neuron ranking | Core | Heuristic readout, not a causal explanation |
| Activation replacement / ablation | Core | MLP outputs; replacement is not a permanent weight edit |
| Direction suppression / column scaling | Core | MLP down projections; exact patch/rollback checks |
| Jacobian Lens | Isolated upstream adapter | Qwen3-0.6B; computes derivatives |
| Transcoder tracing and feature intervention | Isolated upstream adapter | Qwen3-0.6B with released PLTs; computes derivatives |
| ADAG / RelP | Referenced; not integrated | No compatibility claim |
| SAE inspection, path patching, ROME/MEMIT | Not integrated | No working-tool claim |

## Isolated examples

```sh
uv venv .venv-lens --python 3.12
uv pip install --python .venv-lens/bin/python -e . -r examples/lens-requirements.txt
.venv-lens/bin/python examples/jacobian_lens.py

uv venv .venv-tracing --python 3.12
uv pip install --python .venv-tracing/bin/python -r examples/tracing-requirements.txt
PYTHONPATH=src .venv-tracing/bin/python examples/circuit_trace.py
```

Both examples require CUDA and pin upstream Git commits. Preserve each environment's `uv pip freeze` output. The lens fits four short authored prompts and checks finite readouts, native-logit parity, and unchanged weights: this tests integration, not fitted-lens quality. The tracing example checks a finite graph, unchanged logits, and an actual feature intervention.

Qwen3.5's hybrid architecture is separate from Qwen3. Do not apply Qwen3 transcoders to it. Derivative-based results are separate from the gradient-free reproduction.

For high-throughput evaluation install vLLM and this package in another CUDA environment, then use `evaluate --engine vllm`. The adapter maps MLP down-projection tensors, verifies hashes inside workers, and disables/resets prefix caches around edits. Actual GPU verification is required before trusting campaign results; the engine is recorded in every protocol.

Sources: [Jacobian Lens](https://github.com/anthropics/jacobian-lens), [circuit-tracer](https://github.com/decoderesearch/circuit-tracer), [released Qwen3-0.6B transcoders](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0), [ADAG](https://github.com/TransluceAI/circuits), [CircuitKIT](https://arxiv.org/abs/2607.19317).
