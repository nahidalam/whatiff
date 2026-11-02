# Historical implementation notes

Use [the README](../README.md) for current setup and replay instructions.
Local development evidence paths below are historical references, not required
files for running the submitted demo.

**What if your robot saw a different world?**

Controlled counterfactual observations for tabletop action-selection evaluation.

## Run the MVP

The local FastAPI/React application wraps the existing workflow, generators,
isolated Muse Spark policy, independent scoring, and saved evidence. Python 3.9+
and Node.js 18+ are required. Run from the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
npm install
npm run build
.venv/bin/python -m whatiff.web
```

Open **http://127.0.0.1:8790**. Only loopback is bound; no cloud deployment is used.
Port 8790 avoids an existing application on 8787. For frontend development, keep
the API running and use `npm run dev` in a second terminal.

1. The app opens a **Prepared example**, reconstructing deterministic images and
   replaying recorded policy decisions with **zero model calls**.
2. Compare images, expected/observed actions, edit validity, and each contribution.
3. Click **Run live experiment** for fresh, isolated Muse Spark policy decisions,
   live Spark Adversary proposals, and SAM/Pillow controlled occlusion. Only the
   occlusion is executed in controlled mode; the other proposals are saved.
4. Use **Export JSON** to download the evaluation report. Replay and live runs
   are explicitly labeled. Failed live runs can resume saved steps with **Retry**.

The default controlled recipe targets the white mug. The optional **Muse Image ·
3 edits** backend uses the existing three-intervention workflow. Uploaded scenes,
changed tasks, and free-form edits do not inherit the reviewed example's approval;
they remain Inconclusive without separate evidence. No new review UI is included.
API request/response audits, image bytes, criteria snapshots, decisions, and
reports are saved under `data/web/<run-id>/`. Auth remains server-side in `.env`,
which is ignored by Git. Configure it using [.env.example](../.env.example).

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
npm run build
# With the app running and Chrome installed:
npm run test:browser
# Explicit opt-in: incurs live Meta API calls.
env WHATIFF_LIVE_TEST=1 npm run test:browser -- --grep 'live Meta controlled'
```

Browser tests use the installed Google Chrome in an isolated context; set
`CHROME_PATH` if needed. Evidence screenshots and the live browser-exported
report are under `data/browser-evidence/`. The normal browser suite skips live
calls deliberately. Browser and backend tests cover replay, export, responsive
comparison, all outcome labels, retry, criterion integrity, and isolated policies.

Tests use explicit provider doubles and mocked HTTP. Milestone 2 image checks use
synthetic pixel fixtures and real Pillow decoding. Automated tests do **not**
validate live model image quality or robot behavior.

## Live Meta adapter validation

Copy [.env.example](../.env.example) to `.env` (ignored) and configure `MODEL_API_KEY`
locally, or export it in your shell. Only Meta credentials/endpoints are used.

```sh
.venv/bin/python -m whatiff.live --preflight
.venv/bin/python -m whatiff.live --image /absolute/path/to/tabletop.png
```

Preflight checks the authenticated catalog, not inference success. The image
probe saves the original, obtains three Spark proposals, then edits each from
that same original. It does **not** run a policy. Calls incur normal provider
charges; image calls are not retried automatically. Use `--criteria path.json`
for another task/independent criteria. The supplied mug template is unapproved.

Output: `data/meta-probes/<UUID>/probe.json`, `images/`,
`api/<call-id>/request.json`, `response.json`, and `review-template.json`.
Prompts, model settings, images, raw responses, hashes, and live/mock labels are
saved before parsing. Malformed/rejected outputs remain inspectable. HTTP records
retain body/status; auth headers are omitted and credential echoes redacted.
Keep the whole evidence folder together. Default network timeout: 120 seconds.

Retry with the printed UUID and identical image/criteria:

```sh
.venv/bin/python -m whatiff.live --image /absolute/path/to/tabletop.png --resume UUID
```

Saved successful proposals/edits are reused. A call interrupted before checkpoint
may repeat. SAM is off by default; optional grounding can be requested with
`--sam --sam-concept 'white mug'`. Boxes and raw mask tokens (not decoded mask
rasters) are saved. SAM errors are recorded without blocking the core loop.

**Live validation, 2026-10-29:** authenticated preflight confirmed Spark, Muse Image,
and SAM access. On a synthetic tabletop drawing, Spark returned three proposals,
Muse Image returned three reference edits, and SAM localized the mug. The first
Spark attempt exhausted its output budget; the adapter now records explicit low
reasoning effort and a 6,000-token output budget. Automated tests still use mocks.

All three first-run edits returned at 1600×1600 from a 1024×1024 input and were
correctly marked invalid/inconclusive. Visual inspection also found original-mug
movement in the distractor edit. Saved live evidence is in
`data/meta-probes/45b65e1afb1a4ff9bcf763abd3959b8c/`; SAM evidence is under
`data/live-validation/`. This is API smoke testing on synthetic imagery, not a
curated robot-policy benchmark. No policy evaluation or physical execution ran.

A follow-up with an explicitly prepared 1600×1600 input returned three images at
matching dimensions (`data/meta-probes/16b8005e005240dea230cddd06eed9a2/`). Developer
inspection found duplicated occluders and an extra distractor mug; relocation did
not satisfy the independently defined obstruction condition. Diagnostic
outside-region changed-pixel fractions were 1.50%, 7.06%, and 1.03% respectively.
The distractor failed pixel preservation; the other two demonstrate why passing
pixel heuristics alone is insufficient. Diagnostic regions are not curator
approval. All policy-evaluation outcomes remain inconclusive. See the saved
`developer-inspection.json` alongside raw requests, responses, and images.

## Edit validity and preservation

[Muse Image](https://dev.meta.ai/docs/image-generation) supports reference edits
but rejects masks. Prompts request one intervention and preservation of viewpoint,
lighting, geometry, and unrelated objects; compliance is not guaranteed. Requested
dimensions set aspect ratio. Resized outputs are saved and marked invalid, not
silently rescaled. Undecodable/oversized outputs fail generation but remain in the
raw API audit. Inputs must be single-frame PNG/JPEG/WebP, at most 20 MB/16 megapixels,
with normalized EXIF orientation.

`EditValidator` rejects identical pixels or dimension changes. Other edits stay
unverified until independent review is bound to exact image hashes, task,
intervention description, and criterion. Copy `review-template.json` to a separate
file. Inspect both images, then fill reviewer/evidence, normalized `[x1,y1,x2,y2]`
edit regions, and all four semantic checks. Relocation regions should include old
and new locations and intended local shadow changes.

Outside those regions, channel differences above 24 count as changed pixels.
More than 2% changed outside pixels or mean channel difference above 4 invalidates
preservation. Regions covering over half the image cannot establish validity.
These recorded thresholds are conservative heuristics, not calibrated guarantees
of semantic preservation or physical validity.

```sh
.venv/bin/python -m whatiff.live --review-run data/meta-probes/UUID --reviews /absolute/path/to/review.json
```

Review makes no API calls. Only matching, complete curator review plus consistent
pixel checks establishes edit validity. Invalid/unverified edits score
**Inconclusive**. Adapter probes retain an inconclusive evaluation outcome because
no policy was evaluated. A stale review cannot validate different pixels, task,
proposal, or criteria. SAM grounding is supplemental evidence, never ground truth.

## Workflow

### Controlled occlusion and isolated policy (Milestone 3)

The bounded [SAM/Pillow backend](../whatiff/controlled.py) paints one opaque
**sensor-layer** rectangle over about 13% of the target bounding box. It cannot
relocate, duplicate, or modify other objects. No pixels outside the rectangle may
change, even by one channel value. This is observation occlusion, not a generated
physical obstacle. Muse Image remains available as a separate optional backend.

The [independent fixture definition](../examples/controlled-occlusion.json) declares
the invariance criterion before editing or policy calls. Its pickup eligibility
is an image-level fixture assumption; no physical approach or grasp was tested.
Developer visual review confirms the exposed rim/handle and exact preservation,
and a hash-bound criteria snapshot is frozen before policy inference. Arbitrary
uploads and the other intervention templates do not inherit this approval.

Live result: **valid observation intervention**, 5,460 changed pixels inside
`[384,404,436,509)`; **zero changed pixels outside**; one SAM mug match before
and after. Isolated `muse-spark-1.3` selected `PICK_TARGET` on both observations.
Both decisions pass the independently specified action-selection criterion;
this single case establishes neither general robustness nor physical success.

The [policy adapter](../whatiff/policy.py) implements the replaceable `Policy`
contract. Each stateless request contains only one image, the task, and fixed
action definitions. It receives no Adversary prompt, proposal, paired-image
history, edit metadata, review, or expected-action criteria. Policy explanations
are saved but do not determine scores. All live model calls are Meta-only.

Portable replay requires no key or API access:

```sh
.venv/bin/python -m whatiff.demo --example examples/controlled-example.json
```

This regenerates identical edited pixels from the saved SAM box and replays
**recorded**, not freshly inferred, policy decisions. The synthetic original
image and prepared record are included under `examples/`.

For a fresh live pair, run `.venv/bin/python -m whatiff.demo`. It stops at
`awaiting_visual_review` and saves original/edited images, SAM responses, and exact
checks. After inspecting the pair, `whatiff.demo.record_review(folder, reviewer,
evidence)` records independent review; it is not an automatic approval routine.
Then run `.venv/bin/python -m whatiff.demo --evaluate <folder>` for fresh policy
decisions, or `--replay <folder>` for deterministic pixel replay. Policy resumes
reuse completed decisions and reject changed settings, criteria, images or mode.

Recorded live pair and complete policy audit:
`data/controlled-pairs/c5c9b69223da42d1be03570e8a8101fb/`.
The earlier SAM audit for this pair is in the adjacent UUID directory ending
`-api`; future runs keep SAM audit under the pair's `sam-api/` directory.
Only the occlusion pair has independent validity approval. Distractor and
relocation generation issues from Milestone 2 remain unresolved.

An independent policy evaluates the original observation. The Adversary proposes
three interventions. For each, the Generator edits the **original** image, a
validator records evidence, and the same policy evaluates the edit. Deterministic
evaluation compares each action against separately supplied criteria, then ranks
supported violations before inconclusive results and passes.

The Adversary sees only the image and task. Its predictions are stored as
hypotheses and never become ground truth. The policy sees only the image and
task. Provider instances must be configured with the selected intervention types
without receiving expected answers. Expected-action criteria are supplied and
approved by a curator independently of the agents and frozen into the run before
the baseline call. The evaluator cannot modify them.

- `PICK_TARGET`: select the uniquely identifiable target for pickup.
- `REPOSITION`: change viewpoint or approach before attempting pickup.
- `ABSTAIN`: decline because target identity is insufficiently established.

Invariance preserves acceptable actions; adaptation requires different actions.
Changed decisions alone are not failures. Approved criteria and validated edits
are both required for a pass or potential failure; otherwise results are
inconclusive. These are action-selection results. No physical robot execution,
simulation, or grasp success is claimed.

The [tabletop criteria template](../examples/tabletop-criteria.json) defines the mug
scenario separately from the agents. It remains **unapproved** until actual
observations and evidence are curated. Relocation requires repositioning only
when it places the mug behind an existing obstruction; ordinary relocation does
not automatically require a different action.

## Code and persistence

- [Contracts](../whatiff/contracts.py): agent, policy, validator, and optional segmenter interfaces.
- [Controller](../whatiff/workflow.py): creation, bounded execution, checkpoints, and resume.
- [Evaluation](../whatiff/evaluation.py): independent scoring and ranked report.
- [Store](../whatiff/store.py): local JSON records and content-addressed observations.

`RunStore("data/runs")` saves each run under its UUID, with original/generated
image bytes, hashes, task, mode, criteria, non-secret model configuration, raw
policy responses, proposal predictions, edit metadata, validity evidence, and an
activity log. Keep credentials out of configuration and provider output records.
Record actual model IDs, prompt versions and inference settings for live runs.

Create with `Workflow.create(...)`, then execute or retry with
`Workflow.resume(run_id)`. Reconstruct the controller with the same configuration
to resume after restart. Completed steps are reused; a provider call interrupted
before checkpointing may repeat. Each run has one writer; concurrent execution
of the same run is not supported. Failures save the failing step and exception
type; provider exception text is omitted to avoid leaking credentials.

SAM is an optional validation enhancement and cannot block the core loop. Invalid
edits may still receive a policy decision for inspection, but their scores remain
inconclusive. Meta adapters additionally decode and bound images; opaque bytes
are used only by the original controller unit-test doubles.

Run modes explicitly distinguish `live`, `prepared_example`, and `test`.
Saved outputs support replay and inspection; stochastic model regeneration is
not guaranteed to reproduce identical pixels or decisions.

## Remaining limits

The included controlled occlusion is the validated, reproducible demonstration.
Free-form Muse Image edits can resize or modify unrelated content and still need
independent review; the UI correctly leaves them Inconclusive. Uploaded scenes
also require separate expected-action evidence. These are action-selection
experiments, not physical robot task-success measurements.

The [Meta workshop](https://github.com/meta-models/legal-agent-workshop) informed
the saved progress, mandatory verification, and human-review patterns. This
implementation is original code and does not copy workshop source.

## Hackathon runtime constraint

Every model invoked at runtime must be a Meta model. Milestone 3 uses Muse
Spark for the replaceable action-selection policy, replacing the originally
proposed Qwen baseline. The policy remains separate from the three agents and
receives only the image, task, and fixed action definitions—not Adversary prompts,
intervention metadata, or expected answers. Curator-defined ground truth and
deterministic scoring provide correctness evidence; Spark's own judgment of its
decisions is never the sole evidence. No additional agent or infrastructure is
planned. Codex is used for development, outside the application runtime.
