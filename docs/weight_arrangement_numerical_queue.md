# E05 episode numerical queue

`scripts/run_weight_arrangement_queue.py` builds and executes the separate E05 engineering queue. It calls the [two-function validator](weight_arrangement_numerical_validation.md), using one job per source episode. Every selected call, including unavailable calls, remains in that episode job. Each permutation draw is therefore shared across its calls, modalities and targets inside one candidate process.

Queue construction does not launch validation. A completed job means that all declared diagnostic outcomes were recorded. It is not a numerical approval. Finite degenerate maps, numerical failures, and unavailable source contexts remain explicit outcomes; the queue does not select budgets, interpret criteria, or replace failed draws.

## Required protocol and inputs

The builder requires a prospective JSON protocol and its separately supplied SHA-256. The protocol contains:

- `recorded_before_execution: true`, `decision_id`, and `protocol_version`.
- `source_collection_protocol_sha256`, binding the authenticated collector manifest.
- `strata`, each with `id`, `task`, `model`, `bank_sha256`, `checkpoint_mode`, `checkpoint_sha256`, and the complete `checkpoint_identity` object from that bank. Only the declared pretrained and authors checkpoints are supported.
- `validation`, containing every required field of the E05 validator decision. This includes the explicit fixed ladders, repeats, response grid, source runtime settings, deterministic candidate runtime, Torch version, native-registry hash, implementation hashes, E05 scope, parameter seed namespace and master seed, and `parameter_draws`. No draw count or seed is inferred.
- `expected_population`, with exact keys `strata`, `episodes`, `planned_contexts`, `available_contexts`, and `planned_cells`. For 24 planned contexts, 21 available contexts, 12 episodes, six strata and two draws, these values are 6, 12, 24, 21 and 288. These numbers describe that declared roster, not defaults for another study.

The directory of existing banks must contain `<stratum-id>.json`. The builder authenticates each original bank's bytes, regenerates its membership from `<source-root>/<stratum-id>/metrics.jsonl`, and requires exact equality. It copies the bank bytes without rewriting the JSON. It checks both the source collection protocol and the complete checkpoint identity.

The native structural registry remains a separate authenticated input. The runtime validator must match the actual loaded model against that registry. A queue build does not establish runtime model compatibility or numerical validity.

Each derived episode decision retains all `validation` fields, including any separately locked descriptive degeneracy-rule hash. It adds the exact bank, source checkpoint, parent protocol and episode/call membership. The queue does not interpret extra rule fields; a subsequent raw-artifact audit must apply any declared rule.

## Construction and execution

The following is a command template. Paths and hashes must identify the actual locked protocol and existing artifacts.

```text
python scripts/run_weight_arrangement_queue.py build \
  --protocol /protocols/E05.json --protocol-sha256 <trusted-protocol-sha256> \
  --bank-directory /existing-banks --source-root /authenticated-collection \
  --native-registry /protocols/E05-parameter-registry.json \
  --lang-dir /authenticated-language-cache \
  --checkpoint-path /snapshots/<revision>/mp_rank_00_model_states.pt \
  --output /fresh-e05-queue
```

The output directory must not already exist. The named checkpoint path is made absolute without resolving a Hugging Face symlink to its blob basename. Both its bytes and its recorded filename must match. Omit `--checkpoint-path` only when every stratum uses the pretrained checkpoint path authenticated by the source manifest.

The builder prints the queue SHA-256. A worker requires that exact trusted value and an explicit partition:

```text
python scripts/run_weight_arrangement_queue.py worker \
  --queue /fresh-e05-queue/queue.json --queue-sha256 <trusted-queue-sha256> \
  --index 0 --workers 4
```

Jobs use fixed index modulo worker-count assignment. To run only the first episode as a prospectively specified operational measurement, use `--index 0 --workers <number-of-episode-jobs>`. A later normal partition will authenticate and skip that completed episode. This scheduling choice does not remove other planned episodes or permit outcome-dependent selection of which results to retain.

Each job launches two fresh Python processes, `prepare` then `run`. The first authenticates exact source replay and seals the cache. The second receives the preparation completion hash and configures the candidate environment before importing Torch. The queue checks its own bytes, all pinned implementation sources, parent protocol, native registry, episode decision, bank, source metrics, source manifest and any explicit checkpoint before and after each subprocess.

Exclusive claims and log creation prevent concurrent workers from reusing an attempt. Stage completion authenticates exact file membership and all file hashes. The candidate report must contain exactly every planned context by draw by modality by target, with no unfinished or duplicated cell. Completion also binds the retained logs, claims and partial raw files. A previously completed job is skipped only after reauthenticating its inputs and both complete stages.

## Failure and retry boundaries

A child process failure or contract error stops that worker before its next episode. The claim, logs, partial outputs, completed stage artifacts and failure record remain. A scientifically unsuccessful but fully recorded diagnostic cell does not become a process failure merely because it fails a numerical criterion.

The original queue is never automatically retried. An explicit new queue may use `--retry-queue` and `--retry-amendment` together. The amendment must name a nonempty `replaces_jobs` list, state a `reason`, set `recorded_before_retry: true`, and bind `parent_queue_sha256`, the replacement `protocol_sha256`, and the full replacement `replacement_implementation_sha256` map.

Only sealed process failures or never-claimed jobs can enter this execution retry path. Successful diagnostic jobs cannot be replaced. An unsealed claim needs separate interrupted-attempt diagnosis and is refused. Original workers must be stopped before reassigning never-claimed jobs; the worker additionally rechecks that no competing original claim or completion has appeared.

Replacement membership, checkpoint identity, contexts, numerical criteria, seeds and draws must remain identical. Only the parent protocol hash and implementation hash map may differ inside derived decisions. A changed scientific procedure needs a separate prospective protocol and evidentiary interpretation, rather than this execution-repair path.

The retry queue binds the original queue, amendment, every selected prior failure completion, and its retained attempt files. It records unreplaced original job IDs. The full original population counts remain in the new queue, while `jobs` contains only the explicitly replaced attempts. Analysis must combine linked attempts without counting one planned cell twice or treating the smaller retry job list as a new population.

## Validation scope

CPU tests exercise grouped episode membership, source/checkpoint authentication, unavailable calls, preserved checkpoint filenames, separate process arguments, immutable claims, source mutation during execution, completed-artifact verification, full diagnostic-cell accounting, failed-attempt retention and constrained retry lineage. These tests use synthetic collection bridges and subprocesses. They do not establish actual GPU memory cost, model restoration performance, or E05 numerical validity.
