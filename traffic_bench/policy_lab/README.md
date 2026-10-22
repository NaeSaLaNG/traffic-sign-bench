# Policy lab: an LLM agent writes CaRL policies without seeing the rules

An LLM (served locally by vLLM) iterates on a driving policy built on the
pretrained CaRL planner. It works in a sandbox that holds no sign rules, no
benchmark code and none of the authors' rule-based agents, and it learns how
well a policy does only from aggregate metrics.

```
 trusted side (full repo)                         sandbox (host process, harness hooks)
 ─────────────────────────                         ───────────────────────────────
 agent.py ── tool calls ──────────────────────────▶ list/read/write files, run shell
    │  evaluate(policy) → snapshot of the sandbox
    ▼
 judge.py: MetaDrive + SUMO, sign rules, scoring
    │  RemoteCarlPolicy.act()  ── obs (JSON + BEV bytes) ──▶ carl_sandbox/runtime.py
    │                          ◀── {steering, throttle} ───   Policy.act(obs) + CaRL net
    ▼
 per-episode results (private) → per-family rates → agent
```

## What the agent can and cannot see

| visible | hidden |
|---|---|
| `TASK.md`, `carl_sandbox/` (CaRL wrapper, runtime), `check_policy.py`, `policies/` | `traffic_bench/` entirely: `signs/**` (violation logic, zones, thresholds), `eval/**`, `agents/compliance/**`, `*_rule.py`, `curve_aware_idm.py`, `oracle/`, `finetune/`, `scripts/`, docs, READMEs |
| `third_party/CaRL/nuPlan` (upstream CaRL code) | `.git` and its history, `__pycache__`, `*.egg-info`, `data/runs/` |
| the CaRL checkpoint, at `$CARL_CHECKPOINT` | the operator's home directory (HF token, `~/.cursor`, shell history), host env vars, the NFS repo path |
| `plates.txt`: names of the plates that can appear | the network (refused by the harness hooks) |
| per-family rates: `clean_success_rate`, `success_rate`, `no_sign_violation_rate`, `collision_rate`, `out_of_road_rate`, `policy_fault_rate`, `mean_route_completion_pct` | violation timelines, violation texts, GIFs, MP4s, per-episode results, judge logs and fault tracebacks |

How this holds:

* **Allowlist export.** `export_sandbox.py` copies only the template, the CaRL
  code, a standalone `carl_adapter.py` and the plate names, then refuses to
  finish if any text file mentions the benchmark or its rule code.
* **Separate processes.** The policy runs in its own process and talks to the
  simulator over a Unix socket. It receives plain data: the CaRL BEV (stop
  plates are not drawn, and the speed channel comes from map data) plus lists of
  plates, agents and route lanes. It never gets a sign object, so it cannot call
  `check_violation`. Replies are JSON with two finite floats; the judge never
  unpickles anything the policy sends.
* **Trusted orchestrator.** `agent.py` runs the tools. File access resolves
  symlinks and stays inside the sandbox, shell commands run in the sandbox
  directory with a clean environment (no API keys or tokens; `HOME` and
  `TMPDIR` point inside the sandbox), and evaluations run on a snapshot of the
  sandbox. The output hook in `hooks.py` copies only numeric `overall` and
  `by_family` metrics before judge output reaches the builder or critic.
* **Harness hooks.** There is no container: model-written code runs on the
  host as your user. `hooks.py` checks every tool call before it runs and
  refuses, with the reason, instead of executing:
  * shell commands (`run`) with absolute paths outside the sandbox, `..`,
    `~`/`$HOME`, network, package and VCS tools (`curl`, `pip`, `git`...),
    process control (`kill`, `nohup`, `&`), environment dumps and assignments,
    encoded payloads (`base64`, `eval`), or bad Python in `python -c`/heredocs;
  * Python files (`write_file`, and any file the model added or changed,
    checked before every `run`, `evaluate`, `submit` and the gauntlet's gate)
    importing network, subprocess, `ctypes` or `importlib` modules, calling
    `os.system`/`os.kill`/`eval`/`exec`/`__import__`, reaching for
    `__globals__`/`__subclasses__`, or naming host paths in string literals;
  * `sitecustomize.py`, `.pth` files and symlinks leading out of the sandbox.

  A file that fails the checks after a shell command wrote it is moved to
  `<run-dir>/quarantine/` (a changed template file is restored), so it never
  runs. Every refusal goes to `<run-dir>/hooks.jsonl`. Files the sandbox had at
  start are trusted. The hooks are a static check, not a security boundary:
  code written to hide what it does can get past them, so keep the
  orchestrator, the judge and your data out of reach (see Notes) and read the
  transcripts. `--no-harness-hooks` turns them off.
* **Faults are failures.** A crash, timeout (5 s per step), NaN or a failed load
  counts as a failed episode, so a policy cannot drop hard scenes by crashing on
  them.

## Crosswalk policy lab: production cluster run

This is the one-command path for a crosswalk-only gauntlet. The operator runs
it after pulling a revision that contains `run_crosswalk.py`. The Qwen server
already runs separately on the cluster at `http://localhost:8000/v1`; this
command connects to it and never starts or stops it. Use the existing
`andreev-12-env` environment. The checkout must have crosswalk scenes and the
CaRL checkpoint at `checkpoints/carl/nuplan_51479_1B/model_best.pth` (or pass
`--checkpoint`). The working directory must be writable and **outside** the
checkout. Keep secrets and private benchmark data out of the sandbox. With
video output enabled, the environment also needs `ffmpeg` with H.264 support,
or `opencv-python` plus `Pillow` for the MP4 fallback.

```bash
conda activate andreev-12-env
git pull
git submodule update --init third_party/CaRL
curl -fsS http://localhost:8000/v1/models

# Preflight: print the commands and paths without running a simulation.
python -m traffic_bench.policy_lab.run_crosswalk --split valid --iterations 1 --dry-run

# One-round smoke run. Replace the work directory with a writable cluster path.
python -m traffic_bench.policy_lab.run_crosswalk \
  --split valid --iterations 1 --episodes-per-eval 3 \
  --work-dir /home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/scratch/policy_lab \
  --judge-workers 1 --judge-gpus 1

# Full run on the standard train scene pool.
python -m traffic_bench.policy_lab.run_crosswalk \
  --split valid --iterations 10 --episodes-per-eval 60 \
  --work-dir /home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/scratch/policy_lab \
  --judge-workers 4 --judge-gpus 1

# Optional: run the same search on the standard test scene pool.
python -m traffic_bench.policy_lab.run_crosswalk \
  --split test --iterations 10 --episodes-per-eval 60 \
  --work-dir /home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/scratch/policy_lab \
  --judge-workers 4 --judge-gpus 1
```

`--split test` selects the standard test manifest with the same gauntlet loop.
The agent then improves policies using aggregate test scores, so that run is
**not an independent held-out assessment**. The benchmark has no `valid`
scene pool: this command maps `valid` to the standard `train` manifest and
`test` to `test`. It creates a missing standard manifest with
`python -m traffic_bench.eval manifest sign=crosswalk paths.split=<train|test>`;
an existing manifest is used as-is and never overwritten. Use `--manifest`
to supply a prebuilt manifest explicitly.

| Argument | Default | Meaning |
|---|---|---|
| `--split` | `valid` | `valid` → train, `test` → test |
| `--iterations` | `10` | Maximum builder rounds; evaluation budget is N + 1 including baseline; gauntlet may stop early on its normal criteria |
| `--episodes-per-eval` | `60` | Same sampled episodes for every policy in the run |
| `--work-dir` | sibling `policy_lab_runs` | Private sandboxes, logs and results outside the checkout |
| `--run-name` | UTC timestamp | Unique name; an existing run is never overwritten |
| `--llm-url` | `http://localhost:8000/v1` | Existing OpenAI-compatible Qwen server |
| `--model` | auto | Automatically uses the sole model returned by `/v1/models`; required if several models are served |
| `--checkpoint` | `checkpoints/carl/nuplan_51479_1B/model_best.pth` | CaRL weights |
| `--judge-workers`, `--judge-gpus` | `1`, unset | Parallel simulator workers and GPU selection for policy processes |
| `--builder-turns`, `--context-chars`, `--min-delta`, `--no-think` | gauntlet-oriented defaults | Search controls; see `--help` |
| `--no-gifs` | off | Disable GIF and MP4 rendering when speed matters |

Every run writes to `<work-dir>/runs/crosswalk/<split>/<run-name>/`. Watch
`progress.md` while it runs. `final.json` names the submitted champion, whose
code is in `final/sandbox/`. Each evaluation writes `eval_NNN/gifs/*.gif` and
converts each GIF to a neighbouring `.mp4`. `gifs.html` is an operator-facing
video gallery, created at the end; open it through the cluster file browser.
GIFs, MP4s, `eval_NNN/judge/episodes.jsonl`, `judge.log`, `feedback.json`,
`hooks.jsonl`, `evaluation_errors.log`,
quarantine files, transcripts and critic replies remain outside the agent
sandbox. A failed simulator evaluation is reported to the model by a generic
status; inspect the private judge log for the cause.

The builder can list/read/write sandbox files, run checked local commands and
propose a policy. Input hooks reject path escapes, network/package/VCS/process
commands and unsafe Python imports or execution. Before the builder or critic
sees an evaluation, the output hook selects only numeric aggregate rates and
episode counts from `overall` and `by_family`; it drops paths, per-episode
results, stdout/stderr, exceptions and GIFs. Rejected calls are logged in
`hooks.jsonl` and unsafe files are quarantined. The cluster command never
disables these hooks. These are application guardrails, **not a system
sandbox**: model-written Python still runs with the operator's Unix
permissions, so keep sensitive files inaccessible to that account when
possible.

If preflight fails, first check the named missing checkpoint or CaRL submodule,
then that `curl .../v1/models` returns the expected model. If several models
are served, pass `--model` with an exact returned ID. For simulator errors,
inspect `eval_NNN/judge.log`; for rejected agent actions, inspect
`hooks.jsonl`. MP4 conversion failures are recorded in
`eval_NNN/media_errors.log`; the final command exits with an error while
retaining the policy and GIF results. If the run name already exists, choose a
new `--run-name`.

## Running it on the cluster

Pick a writable working directory for sandboxes and runs. It must be
outside the repository (`export_sandbox` refuses a path inside it), so a sibling
of the checkout works well. Run this from the repository root:

```bash
export LAB=/home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/scratch/policy_lab
mkdir -p $LAB
```

1. Install the policy's dependencies into the environment you run the agent
   from; the policy and `check_policy.py` use the same interpreter:

   ```bash
   pip install "numpy<2" gymnasium==0.29.1 opencv-python-headless==4.10.0.84 scipy torch
   ```

2. Export the sandbox outside the repository:

   ```bash
   git submodule update --init third_party/CaRL
   python -m traffic_bench.policy_lab.export_sandbox --out $LAB/sandbox
   ```

3. Check that plain CaRL loads (catches missing pip packages):

   ```bash
   (cd $LAB/sandbox && PYTHONPATH=. CARL_NUPLAN_PATH=third_party/CaRL/nuPlan \
       python check_policy.py policies/carl_baseline.py \
       --checkpoint $OLDPWD/checkpoints/carl/nuplan_51479_1B/model_best.pth)
   ```

4. Serve the model with tool calling. For Qwen-style chat templates:

   ```bash
   vllm serve checkpoints/Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
   ```

5. Run the agent with a **dev** manifest. Keep a disjoint **test** manifest for
   the final score; the agent never sees it.

   ```bash
   python -m traffic_bench.eval manifest sign=crosswalk paths.split=train
   python -m traffic_bench.policy_lab.agent \
      --sandbox $LAB/sandbox --run-dir $LAB/runs/crosswalk \
      --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
      --device cuda \
      --dev-manifest data/runs/crosswalk/train/real_manifest.jsonl \
      --episodes-per-eval 60 --max-evals 20 \
      --llm-url http://<vllm host>:8000/v1 --model Qwen/Qwen2.5-Coder-32B-Instruct
   ```

   `runs/r1/` gets `transcript.jsonl`, `hooks.jsonl` (refused tool calls), one `eval_NNN/` per evaluation (sandbox
   snapshot, private judge output, feedback) and `final/` with the submitted
   sandbox.

6. Score the submission on the held-out manifest:

   ```bash
   python -m traffic_bench.policy_lab.judge --sandbox $LAB/runs/r1/final/sandbox \
       --policy <submitted policy> --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
       --manifest <test manifest.jsonl> \
       --private-dir $LAB/runs/r1/test --out $LAB/runs/r1/test/feedback.json
   ```

   The policy, `check_policy.py` and the agent's `run` tool use the
   orchestrator's interpreter, with the sandbox as working directory,
   `CARL_CHECKPOINT` pointing at the checkpoint and `CUDA_VISIBLE_DEVICES`
   passed through.

## Gauntlet loop

`gauntlet.py` is an alternative to step 5 that splits the work between a
builder, an orchestrator and a critic, so the model that writes a policy never
grades it:

```
 champion (starts as policies/carl_baseline.py, evaluated once)
    │
    ▼  each round
 builder  (fresh conversation; list/read/write/run + propose, no evaluate)
    │  brief: TASK.md, champion + its metrics, critic's biggest gap, past rounds
    ▼  propose(policies/rNN_*.py)
 orchestrator: hooks + check_policy.py ─ fail → back to builder
    │  judge on the same dev episodes (one evaluation)
    ▼
 critic  (fresh, no tools): TASK.md + policies A/B in random order,
    │     source + aggregate metrics, not told which is the champion
    │  → {"winner", "reason", "biggest_gap"}
    ▼
 decision: Δclean > --min-delta (faults not worse) → win; Δ < -min-delta → loss;
           otherwise the critic decides. A win makes the candidate the champion.
```

The critic sees only what the builder may see (code and per-family rates), so
its gap text cannot leak private judge output. The loop stops when the
champion reaches `--target-clean`, when the evaluation budget runs out (the
reference evaluation counts against it), after `--patience` rounds with no new
champion, or after `--max-rounds`.

```bash
vllm serve /home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/traffic-sign-bench/checkpoints/Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --max-model-len 65536 \
  --enable-prefix-caching
```

`--max-model-len` is the server's context window. Above the model's native
length add YaRN, e.g. `--rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'`
(check `max_position_embeddings` in the checkpoint's `config.json` first). If
vLLM refuses to start for lack of KV-cache memory, lower `--max-model-len` or
raise `--gpu-memory-utilization`.

```bash
python -m traffic_bench.policy_lab.gauntlet \
   --sandbox $LAB/sandbox --run-dir $LAB/runs/g1 \
   --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
   --device cuda \
   --dev-manifest data/runs/crosswalk/train/real_manifest.jsonl \
   --episodes-per-eval 60 --max-evals 20 \
   --context-chars 160000 --builder-turns 30 --no-think --save-gifs \
   --llm-url http://<vllm host>:8000/v1 --model Qwen3.8-27B-FP8
```

* `--context-chars` must fit the server: about 3 characters per token, so
  80 000 (the default) for `--max-model-len 32768` and 160 000 for 65536. Once
  the builder transcript passes it, the oldest tool outputs are blanked. A
  request the server still rejects (HTTP 4xx) ends that round instead of being
  retried.
* `--no-think` turns off Qwen3's reasoning block for the builder (the critic
  keeps it). Turns get much faster; policies may get somewhat worse.
* `--builder-turns` caps the tool calls per round (default 60).
* `--judge-workers N` runs each evaluation in N judge processes, each with its
  own simulator and policy process. MetaDrive and SUMO run on the CPU, so this
  is what makes evaluations faster; take about one worker per free core
  (`nproc`) and leave a few for vLLM. `--judge-gpus 2` puts the workers'
  CaRL processes on GPU 2 (`1,2` spreads them over two cards). The feedback is
  the same as with one worker.
* `--save-gifs` writes one GIF per episode to `eval_NNN/gifs/` and converts it
  to MP4 beside the GIF. They stay on the trusted side, but rendering and
  conversion slow each evaluation. `--gif-window` sets the view in metres
  (default 80).

`runs/g1/` gets `progress.md` (live champion and round table),
`gauntlet.jsonl`, `round_NN_builder.jsonl`, `round_NN_critic.json` (the
critic's exact prompt, the A/B mapping and its reply), the `eval_NNN/` dirs,
and `final/sandbox` (the champion's evaluation snapshot), which you score with
step 6 as before. `--critic-url`/`--critic-model` point the critic at another
server or model.

### One sign at a time

To search for the best policy for a single sign, give each sign its own
manifests, sandbox and run dir. The builder writes into the sandbox, so two
signs sharing one would overwrite each other's `policies/rNN_*.py`. One vLLM
server can serve all signs.

1. Build the manifests: train for the search, test for the final score. The
   agent never sees test.

   ```bash
   SIGN=crosswalk
   python -m traffic_bench.eval manifest sign=$SIGN paths.split=train
   python -m traffic_bench.eval manifest sign=$SIGN paths.split=test
   ```

2. Export a sandbox for this sign:

   ```bash
   python -m traffic_bench.policy_lab.export_sandbox --out $LAB/sandbox_$SIGN
   ```

3. Run the gauntlet on the train manifest:

   ```bash
   python -m traffic_bench.policy_lab.gauntlet \
      --sandbox $LAB/sandbox_$SIGN --run-dir $LAB/runs/$SIGN \
      --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
      --device cuda \
      --dev-manifest data/runs/$SIGN/train/real_manifest.jsonl \
      --episodes-per-eval 60 --max-evals 20 --min-delta 0.04 \
      --llm-url http://<vllm host>:8000/v1 --model Qwen3.8-27B-FP8
   ```

   Progress is in `$LAB/runs/$SIGN/progress.md`.

4. Score the champion on the test manifest:

   ```bash
   POLICY=$(python -c "import json;print(json.load(open('$LAB/runs/$SIGN/final.json'))['submitted']['policy_path'])")
   python -m traffic_bench.policy_lab.judge --sandbox $LAB/runs/$SIGN/final/sandbox \
      --policy $POLICY --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
      --device cuda \
      --manifest data/runs/$SIGN/test/real_manifest.jsonl \
      --private-dir $LAB/runs/$SIGN/test --out $LAB/runs/$SIGN/test/feedback.json
   ```

Every evaluation uses the same dev episodes. If the train manifest has fewer
rows than `--episodes-per-eval`, all of them are used. With 60 episodes, one
episode is about 0.017 of `clean_success_rate`, so for a single sign keep
`--min-delta` at 0.03–0.05 so that noise does not change the champion. Signs
can run in parallel as long as each has its own sandbox and run dir.

## Notes and limits

* Model-written code runs with your user's permissions. The hooks refuse
  anything that names a path outside the sandbox, but keep the held-out test
  manifests and anything secret out of `$LAB` and, where you can, run the
  orchestrator as a user that cannot write to the repository or your data.
* **Routes on `dual_path` signs.** The policy cannot change its route. On these
  signs (one-way, no turn, direction) the judge gives a non-`*_rule` policy
  MetaDrive's default route, so the route channel may lead through a branch the
  sign forbids. A route request in the action (for example "avoid road X") is
  the natural extension.
* `NodeRoadNetwork` maps leave the BEV speed channel empty, since there lane
  limits carry the sign's value. The benchmark's SUMO maps use
  `EdgeRoadNetwork`.
* Each evaluation is sequential, at about one policy round trip per 0.1 s of
  simulated time. Keep `--episodes-per-eval` small during search.
