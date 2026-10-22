# Policy lab: an LLM agent writes CaRL policies without seeing the rules

An LLM (served locally by vLLM) iterates on a driving policy built on the
pretrained CaRL planner. It works in a sandbox that holds no sign rules, no
benchmark code and none of the authors' rule-based agents, and it learns how
well a policy does only from aggregate metrics.

```
 trusted side (full repo)                         sandbox (Apptainer, no network)
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
| the CaRL checkpoint, read-only at `/ckpt/carl.pth` | `$HOME` (HF token, `~/.cursor`, shell history), host env vars, the NFS repo path |
| `plates.txt`: names of the plates that can appear | the network (`--network none`) |
| per-family rates: `clean_success_rate`, `success_rate`, `no_sign_violation_rate`, `collision_rate`, `out_of_road_rate`, `policy_fault_rate`, `mean_route_completion_pct` | violation timelines, violation texts, GIFs, per-episode results, judge logs |

How this holds:

* **Allowlist export.** `export_sandbox.py` copies only the template, the CaRL
  code, a standalone `carl_adapter.py` and the plate names, then refuses to
  finish if any text file mentions the benchmark or its rule code.
* **Separate processes.** The policy runs in its own container and talks to the
  simulator over a Unix socket. It receives plain data: the CaRL BEV (stop
  plates are not drawn, and the speed channel comes from map data) plus lists of
  plates, agents and route lanes. It never gets a sign object, so it cannot call
  `check_violation`. Replies are JSON with two finite floats; the judge never
  unpickles anything the policy sends.
* **Trusted orchestrator.** `agent.py` runs the tools. File access resolves
  symlinks and stays inside the sandbox, shell commands run in the container,
  and evaluations run on a snapshot of the sandbox. Judge output is reduced to
  the aggregate JSON before it reaches the model.
* **Faults are failures.** A crash, timeout (5 s per step), NaN or a failed load
  counts as a failed episode, so a policy cannot drop hard scenes by crashing on
  them.

## Running it on the cluster

Pick a writable working directory for images, sandboxes and runs. It must be
outside the repository (`export_sandbox` refuses a path inside it), so a sibling
of the checkout works well. Run this from the repository root:

```bash
export LAB=/home/jovyan/shares/SR006.nfs2/zinkovich/students/sleptsov-and/gavril/scratch/policy_lab
mkdir -p $LAB
```

1. Build the policy image once (it contains no benchmark code):

   ```bash
   apptainer build $LAB/policy.sif traffic_bench/policy_lab/apptainer/policy.def
   ```

2. Export the sandbox outside the repository:

   ```bash
   git submodule update --init third_party/CaRL
   python -m traffic_bench.policy_lab.export_sandbox --out $LAB/sandbox
   ```

3. Check that plain CaRL loads in the container (catches missing pip packages):

   ```bash
   apptainer exec --containall --no-home --bind $LAB/sandbox:/work \
       --bind checkpoints/carl/nuplan_51479_1B/model_best.pth:/ckpt/carl.pth:ro --pwd /work \
       --env PYTHONPATH=/work,CARL_NUPLAN_PATH=/work/third_party/CaRL/nuPlan \
       $LAB/policy.sif python check_policy.py policies/carl_baseline.py
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
      --image $LAB/policy.sif \
      --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
      --gpu --device cuda \
      --dev-manifest data/runs/crosswalk/train/real_manifest.jsonl \
      --episodes-per-eval 60 --max-evals 20 \
      --llm-url http://<vllm host>:8000/v1 --model Qwen/Qwen2.5-Coder-32B-Instruct
   ```

   `runs/r1/` gets `transcript.jsonl`, one `eval_NNN/` per evaluation (sandbox
   snapshot, private judge output, feedback) and `final/` with the submitted
   sandbox.

6. Score the submission on the held-out manifest:

   ```bash
   python -m traffic_bench.policy_lab.judge --sandbox $LAB/runs/r1/final/sandbox \
       --policy <submitted policy> --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
       --image $LAB/policy.sif --manifest <test manifest.jsonl> \
       --private-dir $LAB/runs/r1/test --out $LAB/runs/r1/test/feedback.json
   ```

   `--launcher local` runs the policy without a container. It is for
   debugging only and gives no isolation. `agent.py` and `gauntlet.py`
   accept it too (then `--image` is not needed): the policy, `check_policy.py`
   and the agent's `run` tool use the orchestrator's interpreter on the host,
   with the sandbox as working directory, `CARL_CHECKPOINT` pointing at the
   checkpoint and `CUDA_VISIBLE_DEVICES` passed through. Model-written code can
   then read and write anything you can, and reach the network. Use it only
   where Apptainer cannot run (e.g. pods that forbid mount namespaces), and
   watch the run.

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
 orchestrator: check_policy.py in the container ─ fail → back to builder
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
  --reasoning-parser qwen3
```

```bash
python -m traffic_bench.policy_lab.gauntlet \
   --sandbox $LAB/sandbox --run-dir $LAB/runs/g1 \
   --image $LAB/policy.sif \
   --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
   --gpu --device cuda \
   --dev-manifest data/runs/crosswalk/train/real_manifest.jsonl \
   --episodes-per-eval 60 --max-evals 20 \
   --llm-url http://<vllm host>:8000/v1 --model Qwen3.8-27B-FP8
```

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
      --image $LAB/policy.sif \
      --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
      --gpu --device cuda \
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
      --image $LAB/policy.sif --gpu --device cuda \
      --manifest data/runs/$SIGN/test/real_manifest.jsonl \
      --private-dir $LAB/runs/$SIGN/test --out $LAB/runs/$SIGN/test/feedback.json
   ```

Every evaluation uses the same dev episodes. If the train manifest has fewer
rows than `--episodes-per-eval`, all of them are used. With 60 episodes, one
episode is about 0.017 of `clean_success_rate`, so for a single sign keep
`--min-delta` at 0.03–0.05 so that noise does not change the champion. Signs
can run in parallel as long as each has its own sandbox and run dir.

## Notes and limits

* If `--network none` is not allowed for unprivileged users on the cluster,
  `--allow-network` drops it; then block egress some other way.
* Run the orchestrator, the judge and vLLM as a user whose files the model
  never touches; the model reaches the host only through `agent.py`.
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