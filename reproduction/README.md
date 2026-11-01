# MyST-Fusion: M7 reproducibility materials

This directory is a standalone, minimal distribution for Reviewer 2, M7:
implementation, processed entity types, mined rules, and evaluation scripts.
Copy the whole directory to an empty location. All commands below run from this
directory; paths passed to `scripts/run.py` are relative to this directory.

| M7 item | Included material |
| --- | --- |
| Implementation | `code/myst_fusion/`: entity typing and dataset-local standardization, training type frequencies, rule discovery/confidence, directed grounding, bounded-P fusion, complete-chain prompts, decoding and evaluation |
| Processed type labels | `assets/<dataset>/entity2label_clustered_final.json`, with the fixed entity and base-relation ID maps for all five datasets |
| Mined rule sets | `assets/<dataset>/rules/joint.json`; target datasets also include `uniform.json`, `time.json`, `type.json`; training type counts and percentages are supplied |
| Evaluation scripts | `scripts/run.py predict` saves the complete searched entity pool; `metrics` replays filtering and Hits without a model |

`configs/` contains the current response parameters, benchmark checksums and one
shared training configuration. `assets/provenance.json` records each original
repository path, SHA-256 and available mining parameters. `MANIFEST.json` covers
all distributed files except itself. Model weights, raw benchmark facts, prepared
caches, prediction logs and historical analysis directories are excluded.
`configs/benchmark_source.json` records the public source-file hashes verified
on 2026-10-31, including the inputs to the explicit WIKI conversion.

## Protocol and asset provenance

The implementation follows current `response.md`: nondecreasing body times with
all body facts strictly before the head/query; independent uniform body-confidence
sampling; normalized time/type transition products; complete directed grounding;
same-chain fusion before endpoint maximization; shared-fact memberships; whole-chain
packing; and two-step 100-by-100 joint-probability GID decoding.

For recurrence `n` and query-relation endpoint-type frequency `F`, use
`P = normalize(normalize(n) * (1 + beta * normalize(F)))`. Normalization is by
the maximum over the full query endpoint/chain pool, with a zero channel kept at
zero. Recurrence counts visible fact occurrences across historical relations in
the prediction direction, independently of rule-chain multiplicity. Temporal
recency is averaged across all chain edges, using the oldest permitted past fact
as the common scale. Logical support is the maximum fully matching rule confidence.
Only explicitly repeated variables require equal entity bindings.

| Dataset | alpha | lambda | beta | Supplied main rules |
| --- | ---: | ---: | ---: | ---: |
| ICEWS14 | 1 | 1 | 1 | 32,126 |
| ICEWS18 | 1 | 1 | 1 | 42,606 |
| YAGO | 0.05 | 1 | 0.25 | 96 |
| GDELT | 1 | 1 | 1 | 88,633 |
| WIKI | 1 | 1 | 1 | 497 |

Target coefficients are transcribed from the current response, not the archived
v1 weight manifest. The author confirmed on 2026-10-31 that GDELT/WIKI source
training uses equal T/P/L coefficients (1,1,1) and beta=1. This source setting is
recorded in `assets/provenance.json`. The endpoint budget is 50, maximum body length 3 and context budget
1536, including the longest mapped GID and EOS. Fixed benchmark IDs are retained;
WIKI's supplied entity mapping is sparse and remains fixed; the public-to-fixed
alias conversion is described below. Archived type patterns use
percentages; the loader divides by 100 to implement Equation 11's fractions.

**The supplied type, pattern and rule assets are unmodified.** The target control banks
come from `analysis/rule_walk_ablation_rerun_n200`; their recorded missing-pattern
weight is 1.0 on the percentage scale. The historical main miner also defaults to
1.0 on that scale, while the main bank files do not themselves record this parameter.
Since observed weights are `100*F`, this default is equivalent to `epsilon_type=0.01`
on the frequency scale after normalization. New mining here uses that frequency
scale and a single-worker Python RNG. It is deterministic
for this implementation, but does not reproduce historical NumPy/worker streams.
Use `prepare --rules` to evaluate a newly mined bank under a new output name.

The author confirmed on 2026-10-31 that the supplied final types, training patterns
and rule banks are the same files used in the external revised experiments.
`assets/provenance.json` records this confirmation and the exact file hashes.
This package supplies the revised protocol implementation and confirmed assets;
packaging and CPU checks do not constitute a new run of model training or benchmark Hits.

## 1. Environment and checks

Use Python 3.10 or 3.11 for LlamaFactory 0.9.1 and a CUDA-compatible PyTorch build
for the intended machine. Core reference checks and asset verification use only
the Python standard library.

```bash
python -m pip install -r requirements.txt
python scripts/verify.py
python scripts/check_core.py
python scripts/check_decoder.py
```

`check_decoder.py` requires PyTorch and exercises CPU synthetic logits, including
full-vocabulary normalization, both Top-100 pruning steps and joint-score ties.
It does not load a checkpoint. `check_core.py` skips its optional native comparison
until the extension is built; its remaining checks require no model dependencies.

The PyTorch extra in `requirements.txt` is required: LlamaFactory 0.9.1 lists torch
as an optional installation extra. Select a CPU/CUDA wheel for your machine using
the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).

For full-dataset additive TPL preparation, build the included optional accelerator:

```bash
python scripts/build_native.py build_ext --inplace
python scripts/check_core.py
```

Linux needs Python development headers and a C++17 compiler; Windows needs the
Visual C++ build tools. Build with the same Python interpreter used for preparation.
The default `--backend python` exhaustively enumerates witnesses and is intended
for small checks. Use `--backend native` for full additive TPL runs. Other channel
combinations and max/mul use exhaustive enumeration even when native is selected,
so zero-channel ties retain the same exact witness-selection semantics.

## 2. Obtain and verify benchmark facts

Obtain ICEWS14, ICEWS18, YAGO, GDELT and WIKI from the data source cited in M7:
[TKG-forecasting-datasets](https://github.com/waltbai/TKG-forecasting-datasets).
Its processed split format is subject/relation/object/time, tab-separated.
Supply each dataset directory containing `train.txt`, `valid.txt`, `test.txt`:

```bash
python scripts/run.py import-data --dataset ICEWS14 --source /datasets/ICEWS14
python scripts/run.py import-data --dataset ICEWS18 --source /datasets/ICEWS18
python scripts/run.py import-data --dataset YAGO --source /datasets/YAGO
python scripts/run.py import-data --dataset GDELT --source /datasets/GDELT
python scripts/run.py import-data --dataset WIKI --source /datasets/WIKI
python scripts/verify.py --data-root data
python scripts/check_paper_cases.py --data-root data
```

The import uses supplied ID maps, removes an optional legacy fifth column and
normalizes whitespace/newlines. For WIKI it first applies the explicit alias-ID
conversion below. It checks **every converted fact and row order** against
`configs/benchmark_checksums.json` before writing. That file contains original
byte hashes, LF hashes, canonical four-column hashes and split sizes; timestamps
and IDs cannot silently change beyond the declared WIKI conversion.
`verify --data-root` additionally recomputes the
supplied type-pattern counts using training facts only.

**WIKI preprocessing:** the public benchmark has 12,554 IDs, while the existing
experiment mapping has 12,513 distinct mapped entities, as stated in response
Appendix B. `assets/WIKI/entity_id_redirects.json` explicitly maps the 41 removed
alias IDs to the retained fixed IDs (40 aliases occur in the facts; one is unused).
Import applies this map once to subject/object IDs, including when using `--format
regcn`. It changes no relation, time, row order, split size, or observation count.
The three converted public splits match the supplied experimental checksums exactly;
the original public file hashes and conversion details are in `assets/provenance.json`.
Already-converted input is unchanged by this map. The imported fixed IDs are then
used throughout training, retrieval and prediction; no query-local renumbering occurs.

`check_paper_cases.py` requires the built native extension (or `--backend python`)
and recomputes three previously documented historical product-form P-variant
multi-hop witnesses, their normalized channel scores and retrieval ranks
28/11/15 directly from the supplied assets and benchmark history. These are
structural path checks, not the current Table 10 full-method Coverage@50 results.
It does not use a prediction model and does not treat retrieval ranks as model Hits.

For raw RE-GCN facts, pass `--format regcn`: subtract one from ICEWS14 time IDs,
divide ICEWS18 times by 24 and GDELT times by 15, leave YAGO/WIKI times unchanged,
and remove the fifth column. Names come from the supplied maps. A split/hash mismatch
stops the import rather than attempting to infer another benchmark variant.

## 3. Prepare prompts

Obtain the public `meta-llama/Meta-Llama-3-8B` checkpoint and associated tokenizer
separately and place them in `models/llama3_8b` (or pass an absolute local path).
Preparation uses the actual tokenizer, audits all GIDs for at most two ID tokens,
and reserves answer space without consulting the gold answer.

```bash
python scripts/run.py prepare --dataset GDELT --split train --tokenizer models/llama3_8b --backend native --output processed/GDELT-train.json
python scripts/run.py prepare --dataset WIKI --split train --tokenizer models/llama3_8b --backend native --output processed/WIKI-train.json
python scripts/run.py prepare --dataset ICEWS14 --split train --tokenizer models/llama3_8b --backend native --output processed/ICEWS14-train.json
python scripts/run.py prepare --dataset ICEWS14 --split test --tokenizer models/llama3_8b --backend native --output processed/ICEWS14-test.json
```

Source training samples 100,000 GDELT and 30,000 WIKI head/tail examples with seed
42. Target preparation includes every head/tail query; replace ICEWS14 with ICEWS18
or YAGO for the other targets. Use `--split valid` for validation. Training history
uses training facts; validation additionally sees strictly earlier validation facts;
test additionally sees strictly earlier test facts. Filtering labels are separate
and use other true answers at the exact query timestamp.

Preparation produces SFT JSON, `dataset_info.json` and metadata with input/code/
tokenizer hashes, query counts and pre-/post-packing endpoints. Changed caches fail
validation; use a new output name or intentional `--overwrite`. `--max-samples 20`
is a small workflow check and is recorded as a sample, not a full benchmark result.

For existing control banks use `--variant uniform`, `time`, or `type`. Channel
comparisons use `--channels T/P/L/TP/TL/PL/TPL` with the retained full-model weights.
The P-only definition controls in Table 4 use `--channels P` together with
`--p-variant type_frequency`, `recurrence`, or `product`; the main definition is
`--p-variant bounded` (default). All variants retain the same chain pool and budget.
Fusion controls use `--operator max/mul` without alpha/lambda multipliers; repeated
shared-fact serialization uses `--repeat-shared-facts`. Training-semantic controls
use `--omit entity relation type` as needed **only when preparing training examples**;
inference keeps full semantics. Store variants in separate prepared directories.

## 4. Train and predict in the three settings

The shared training template uses BF16, LoRA rank/alpha/dropout 8/16/0, all linear
modules, one epoch, LR 1e-4, device batch 32 and accumulation 16. The empty SFT
template trains only answer tokens. Run the source training once:

```bash
python scripts/run.py train --setting source-only --model models/llama3_8b --output outputs/source --config-output outputs/train-source.yaml
python scripts/run.py train --setting target-only --dataset ICEWS14 --model models/llama3_8b --output outputs/ICEWS14/target-only --config-output outputs/train-ICEWS14-target.yaml
python scripts/run.py train --setting source-to-target --dataset ICEWS14 --model models/llama3_8b --source-adapter outputs/source --output outputs/ICEWS14/source-to-target --config-output outputs/train-ICEWS14-transfer.yaml
```

Use `--write-only` to generate and inspect the YAML without launching training.
Source-to-target continues training the source LoRA adapter on target examples;
target-only starts from the public base. Repeat target commands for ICEWS18/YAGO.
For semantic controls pass the corresponding `--prepared-dir` and train a separate
checkpoint. Source-only forecasting uses target-local evidence with the source adapter:

```bash
python scripts/run.py predict --prepared processed/ICEWS14-test.json --model models/llama3_8b --adapter outputs/source --output outputs/ICEWS14-source-only.jsonl
python scripts/run.py predict --prepared processed/ICEWS14-test.json --model models/llama3_8b --adapter outputs/ICEWS14/target-only --output outputs/ICEWS14-target-only.jsonl
python scripts/run.py predict --prepared processed/ICEWS14-test.json --model models/llama3_8b --adapter outputs/ICEWS14/source-to-target --output outputs/ICEWS14-source-to-target.jsonl
```

Predictions retain every distinct valid GID found in the 100-by-100 searched pool,
including GIDs absent from the prompt. Both distributions are normalized over the
full vocabulary before pruning. EOS/nonnumeric second-token termination retains
that token's conditional probability. Duplicate GIDs keep their maximum joint
probability, with ascending token-ID-pair ties. No final Top-10 truncation occurs
before evaluation. Each JSONL row includes `query_id`, `candidate_sequences`,
`scoring_protocol` and `prepared_sha256`; the sidecar records the model, tokenizer
and adapter fingerprints. Completed JSONL rows remain replayable if inference stops.

## 5. Replay metrics without a model

```bash
python scripts/run.py metrics --prepared processed/ICEWS14-test.json --predictions outputs/ICEWS14-source-only.jsonl --output outputs/ICEWS14-source-only-metrics.json
```

The complete prepared query set defines the denominator. Missing predictions and
empty candidate pools are misses; duplicated or unexpected query IDs and incompatible
fingerprints fail explicitly. Other true answers at the same timestamp are removed
from the **complete searched pool**, retaining the current gold answer, before
ranking and selecting the ten highest joint scores. Output includes raw/filtered
Hits@1/3/10, per-query ranks, the denominator, missing prediction count and sample/
complete scope. Retrieval Coverage@50 is not a prediction metric.

## Optional regeneration

Use new output paths to preserve the supplied assets:

```bash
python scripts/run.py types --dataset YAGO --base-url http://localhost:8000/v1 --output outputs/new-types/YAGO
python scripts/run.py patterns --dataset YAGO --output outputs/new-patterns/YAGO
python scripts/run.py mine --dataset YAGO --variant joint --output outputs/new-rules/YAGO-joint.json
python scripts/run.py prepare --dataset YAGO --split test --tokenizer models/llama3_8b --rules outputs/new-rules/YAGO-joint.json --backend native --output processed/new-YAGO-test.json
```

Typing uses a separately started OpenAI-compatible service at temperature zero.
Configure that service with the same Meta-Llama-3-8B checkpoint/tokenizer and BF16
precision specified in the response;
`MYST_TYPING_API_KEY` is read from the environment. Standardization is dataset-local.
Regenerated type files are separate outputs; the supplied final labels remain the
inputs to default pattern/mining commands. New mining uses training facts only,
lengths 1--3, 200 walks per relation/length, seed 123, epsilon 0.01 and 500 independent
uniform body attempts per discovered rule. Failed/duplicate attempts consume the
budget; unsupported valid bodies remain in confidence denominators and multiple
later heads count once. Revalidate coefficients before reporting results with new
types or rules; the supplied target coefficients are the response configuration,
not a validation certificate for a regenerated bank.

## Prepare the accompanying archive

From this directory, create the submission archive from the verified file manifest:

```bash
python scripts/package.py --output ../build/MRM-M7-reproduction.zip
```

The archive contains only the files listed in `MANIFEST.json` and the manifest
itself, with every byte hash verified after compression. Compiled extensions,
build directories, models, caches, outputs and internal analysis records are excluded.
