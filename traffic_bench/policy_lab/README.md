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

1. Build the policy image once (it contains no benchmark code):

   ```bash
   apptainer build /scratch/policy_lab/policy.sif traffic_bench/policy_lab/apptainer/policy.def
   ```

2. Export the sandbox outside the repository:

   ```bash
   git submodule update --init third_party/CaRL
   python -m traffic_bench.policy_lab.export_sandbox --out /scratch/policy_lab/sandbox
   ```

3. Check that plain CaRL loads in the container (catches missing pip packages):

   ```bash
   apptainer exec --containall --no-home --bind /scratch/policy_lab/sandbox:/work \
       --bind checkpoints/carl/nuplan_51479_1B/model_best.pth:/ckpt/carl.pth:ro --pwd /work \
       --env PYTHONPATH=/work,CARL_NUPLAN_PATH=/work/third_party/CaRL/nuPlan \
       /scratch/policy_lab/policy.sif python check_policy.py policies/carl_baseline.py
   ```

4. Serve the model with tool calling. For Qwen-style chat templates:

   ```bash
   vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
   ```

5. Run the agent with a **dev** manifest. Keep a disjoint **test** manifest for
   the final score; the agent never sees it.

   ```bash
   python -m traffic_bench.policy_lab.agent \
       --sandbox /scratch/policy_lab/sandbox --run-dir /scratch/policy_lab/runs/r1 \
       --image /scratch/policy_lab/policy.sif \
       --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
       --dev-manifest <dev manifest.jsonl> --episodes-per-eval 60 --max-evals 20 \
       --llm-url http://<vllm host>:8000/v1 --model Qwen/Qwen2.5-Coder-32B-Instruct
   ```

   `runs/r1/` gets `transcript.jsonl`, one `eval_NNN/` per evaluation (sandbox
   snapshot, private judge output, feedback) and `final/` with the submitted
   sandbox.

6. Score the submission on the held-out manifest:

   ```bash
   python -m traffic_bench.policy_lab.judge --sandbox /scratch/policy_lab/runs/r1/final/sandbox \
       --policy <submitted policy> --checkpoint checkpoints/carl/nuplan_51479_1B/model_best.pth \
       --image /scratch/policy_lab/policy.sif --manifest <test manifest.jsonl> \
       --private-dir /scratch/policy_lab/runs/r1/test --out /scratch/policy_lab/runs/r1/test/feedback.json
   ```

   `--launcher local` runs the policy without a container. It is for
   debugging only and gives no isolation.

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
