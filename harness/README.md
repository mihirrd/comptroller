# Comptroller SWE-bench Lite harness

This directory holds tooling and docs for running **SWE-bench Lite** (`princeton-nlp/SWE-bench_Lite`) with **Comptroller**: evaluation smoke tests, inference CLIs, and optional Docker materialization of `/testbed`.

## 1. Pin and install the `swebench` package (SWE-bench repo)

The Python module you import and run is `swebench`; it ships from **[princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench)** (install from Git, not a standalone PyPI name).

**Pinned ref:** see [`SWEBENCH_GIT_REF`](SWEBENCH_GIT_REF) (currently `v4.1.0`). Bump that file when you intentionally upgrade.

### Option A — editable install (recommended for debugging)

```bash
export SWEBENCH_REF="$(tr -d ' \n' < SWEBENCH_GIT_REF)"
git clone --depth 1 --branch "${SWEBENCH_REF}" https://github.com/princeton-nlp/SWE-bench.git ../SWE-bench
cd ../SWE-bench
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
pip install "datasets>=3.0.0"
```

### Option B — `uv` in `harness/` (Comptroller 3.13 venv)

From **`harness/`**:

```bash
uv venv -p 3.13
source .venv/bin/activate
uv sync
```

Install SWE-bench into **this or another** venv (eval may use Python 3.11 if you prefer):

```bash
export SWEBENCH_REF="$(tr -d ' \n' < SWEBENCH_GIT_REF)"
uv pip install "git+https://github.com/princeton-nlp/SWE-bench.git@${SWEBENCH_REF}"
# Or from a local clone:
# uv pip install -e ../SWE-bench
```

Verify:

```bash
python -c "import swebench; print('swebench OK')"
```

## 2. Prerequisites for evaluation

- **Docker** installed and running (SWE-bench pulls/builds instance images).
- **Disk and RAM** sufficient for images; first-time **environment** image builds run **conda** inside the container and can **exit 137 (OOM)** if Docker’s memory limit is too low (see §7).
- **Network** for Hugging Face and image pulls.

## 3. Pick a valid Lite `instance_id`

Gold smoke tests must use an id that exists in **`princeton-nlp/SWE-bench_Lite`**.

With `datasets` installed:

```bash
cd harness
source .venv/bin/activate   # if using harness/.venv
python scripts/print_one_lite_instance_id.py
```

Use that id (or any row from the dataset) in the next step.

## 4. Smoke test: `run_evaluation` with **gold** on one Lite instance

This checks Docker + `swebench` + Lite wiring. It does **not** run Comptroller.

```bash
export INSTANCE_ID="paste_instance_id_here"
chmod +x scripts/smoke_eval_lite_gold.sh
./scripts/smoke_eval_lite_gold.sh
```

Or directly:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path gold \
  --instance_ids "${INSTANCE_ID}" \
  --max_workers 1 \
  --run_id validate-gold-lite-smoke
```

Gold patches should **resolve** the instance. Logs and reports depend on your pinned `swebench` version (console output and under `logs/` when cwd is relevant).

## 5. Predictions JSONL format

Each line is typically:

```json
{
  "instance_id": "owner__repo-issue",
  "model_name_or_path": "comptroller",
  "model_patch": "unified diff string"
}
```

Confirm field names against your installed `swebench` ([evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)).

## 6. Inference CLIs

**Requires Python 3.13+** (same as Comptroller).

```bash
cd harness
uv venv -p 3.13
source .venv/bin/activate
uv sync
# LiteLLM keys, e.g. OPENAI_API_KEY — use `.env` in cwd or parents (via `python-dotenv` from Comptroller)
```

### `comptroller-swebench` (existing checkout or Docker materialize)

Runs Comptroller on selected dataset rows and **appends** JSONL lines.

With a **git checkout** you prepared at the instance **base commit**:

```bash
comptroller-swebench \
  --workspace-root /path/to/checkout \
  --instance-ids astropy__astropy-12907 \
  --output predictions.jsonl \
  --model gpt-4o \
  --max-tokens 500000
```

First **N** rows (same workspace caveat: normally one checkout per instance):

```bash
comptroller-swebench --workspace-root /path/to/repo --limit 1 --output predictions.jsonl
```

**Docker materialize** (optional): `--materialize-docker` and related flags — see **§7**.

Module: `python -m comptroller_swebench ...` (same flags as the console script).

### `comptroller-swebench-clone-run` (clone → infer → delete clone)

For a **single** `instance_id`: load the Lite row, **`git clone`** `https://github.com/{repo}` at **`base_commit`** into a **temp directory**, run Comptroller, **append** one JSONL line to **`--output`**, then **remove** the temp tree. Needs **`git`** on `PATH` and GitHub access. Does **not** use Docker.

```bash
comptroller-swebench-clone-run astropy__astropy-12907 \
  --output predictions.jsonl \
  --model gpt-4o \
  --max-tokens 500000
```

- **`--scratch-parent`**: parent directory for the temp clone (default: system temp).
- On failure after the instance was found, an **empty `model_patch`** line may still be appended (same spirit as the main CLI).

Module: `python -m comptroller_swebench.clone_workflow INSTANCE_ID ...`.

### `uv` cache

If `uv sync` fails on wheel metadata under a system temp cache:

```bash
export UV_CACHE_DIR="$(pwd)/.uv-cache"
```

That path is ignored by git via `harness/.gitignore`.

## 7. Optional: materialize `/testbed` from Docker

SWE-bench images use **`/testbed`** as the repo root (`DOCKER_WORKDIR`). **`--materialize-docker`** builds/pulls images and exports that tree to disk for host-side runs.

1. Install **`swebench`** in the venv you use for this CLI (§1); keep **`SWEBENCH_GIT_REF`** aligned with `run_evaluation`.
2. **Docker** must have enough **RAM** for image builds. First materialize may build **base + environment + instance** layers. **`linux/x86_64`** matches upstream defaults (on Apple Silicon, Docker often **emulates amd64**, which is slower and more memory-hungry). **Exit code 137** during `setup_env.sh` usually means **OOM** — raise Docker Desktop **Settings → Resources → Memory**, free host RAM, retry. Check **`harness/logs/build_images/env/.../build_image.log`**.
3. Use **`--materialize-docker`** with `comptroller-swebench`.

Behavior:

- **One instance:** `--workspace-root` is the export directory (created if needed).
- **Several instances:** `--workspace-root` is a **parent**; each instance goes to `<workspace-root>/<instance_id>/`.

Other flags: **`--overwrite-materialized`**, **`--force-rebuild-docker-image`**, **`--docker-run-id`**.

SWE-bench log paths are often relative to **process cwd**; the materialize path **chdirs** into `harness/` while talking to Docker so logs stay under **`harness/logs/`** (including `logs/materialize/...`). Exported files may be **root-owned** from the image; `chown -R` locally if needed.

## Scoring model output

After `predictions.jsonl` exists, run **`run_evaluation`** (§4) with **`--predictions_path`** pointing at that file instead of `gold`.

## Recovering this file from git

If you had committed `harness/README.md`:

```bash
git checkout HEAD -- harness/README.md
# or from an older commit:
git show <commit>:harness/README.md > harness/README.md
```

`git reflog` helps find commits if you only soft-reset.

---

Keep **`SWEBENCH_GIT_REF`** in sync with the `swebench` version you use for scoring so inference and evaluation stay aligned.
