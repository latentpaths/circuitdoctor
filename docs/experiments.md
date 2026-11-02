# Experiment protocol

Question: can an open editor produce a persistent repair that improves held-out behavior, and do internal measurements help beyond behavioral search and simpler deployment changes?

## Frozen conditions

`configs/experiment.json` specifies seeds 0/1/2, GPT-OSS 20B/120B editors, 25 diagnostic batches, and ten patches per run. The target config pins Qwen3.5-0.8B and the report's thinking-mode sampling: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 output tokens.

Seeded sampling without replacement selects distinct development and validation questions from GSM8K train. Final tests contain 256 GSM8K test questions, 128 MATH-500 questions, and all 541 IFEval prompts. The manifest stores IDs, dataset fingerprints, and content hashes. The final test and published solution are inaccessible through investigator tools.

## Controls and selection

Compare unchanged weights, the published patch, scripted contrast search, random interventions, and both editor sizes with and without internal measurements. Run searches for all three seeds. Simple baselines are presence penalty 2.0 and non-thinking mode (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, 8,192 tokens).

Use equal diagnostic/candidate ceilings and also show actual sequences, tokens, time, and dollars. A tool call is not a uniform compute unit. Internal access permits data-derived directions, so interpret that comparison as the value of the instrumented workflow, not pure editor intelligence.

Freeze candidates before final evaluation. Select positive validation accuracy gains; break ties by fewer token-limit hits, then fewer changed entries. Retain unchanged weights when no candidate improves validation. Report every seed, not just the best. Prior knowledge of published results cannot be excluded; call these rediscovery attempts.

## Measurements and claims

Report accuracy, truncation, repetition, generated/prompt tokens, runtime, edit size, fixed/broken answers, and paired bootstrap intervals. Resample by problem, retaining repeated samples within problems.

Success requires at least +10 GSM8K percentage points with a positive lower 95% bound, at least 50% fewer GSM8K token-limit hits, and at most a 2-point IFEval decline. Also compare with the strongest simple baseline. Neither shorter answers nor a local intervention proves improved underlying mathematical ability or a complete circuit.

The original full protocol uses MATH-500 at four samples/problem, GSM8K at two, and IFEval at two. Expand only if measured throughput with a 25% time margin fits the remaining reserved GPU time and budget. Subset results are never numerical parity with the paper.

Every request has a prior cost reservation; unknown outcomes retain it. Evaluation checkpoints contain only completed batches and bind to immutable protocol hashes. Changed configurations require new run directories. Preserve incomplete evidence; distinguish execution failure from model failure.
