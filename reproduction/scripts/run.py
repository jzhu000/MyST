"""Run from any working directory; relative paths are rooted at this package."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from myst_fusion.data import DATASETS, TARGETS, Dataset, digest, read_facts, read_json, validate_facts, write_json, apply_entity_redirects
from myst_fusion.workflow import canonical_hash, check_benchmark, load_prepared, prepare, replay


def tokenizer_at(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def import_data(args):
    dataset = Dataset(ROOT, args.dataset)
    converted = {}
    for split in ("train", "valid", "test"):
        facts = read_facts(Path(args.source) / (split + ".txt"))
        if args.format == "regcn":
            def timestamp(t):
                return t - 1 if args.dataset == "ICEWS14" else t // 24 if args.dataset == "ICEWS18" else t // 15 if args.dataset == "GDELT" else t
            facts = [(h, r, o, timestamp(t)) for h, r, o, t in facts]
        facts = apply_entity_redirects(facts, dataset.entity_redirects)
        validate_facts(facts, dataset.entities, dataset.relations)
        check_benchmark(ROOT, args.dataset, split, facts)
        converted[split] = facts
    folder = Path(args.data_root) / args.dataset
    if any((folder / (s + ".txt")).exists() for s in converted):
        raise FileExistsError("Destination already contains benchmark facts; choose an empty data root")
    folder.mkdir(parents=True, exist_ok=True)
    for split, facts in converted.items():
        (folder / (split + ".txt")).write_text("".join("\t".join(map(str, f)) + "\n" for f in facts), encoding="utf-8")
    write_json(folder / "import.json", {"dataset": args.dataset, "format": args.format,
               "source": str(Path(args.source).resolve()), "declared_entity_redirects": len(dataset.entity_redirects), "splits": {
                   s: {"rows": len(f), "canonical_sha256": canonical_hash(f)} for s, f in converted.items()}})
    print(f"Imported and verified {args.dataset}: " + str({s: len(f) for s, f in converted.items()}))


def regenerate(args):
    from myst_fusion.entity_types import completion, infer_types, type_frequencies
    from myst_fusion.mining import RuleMiner
    data = Dataset(ROOT, args.dataset, args.data_root)
    training = None
    if args.command != "types":
        training = read_facts(data.data_dir / "train.txt")
        validate_facts(training, data.entities, data.relations)
        check_benchmark(ROOT, args.dataset, "train", training)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Use a new output path; supplied assets are never overwritten")
    if args.command == "types":
        raw, mapping, final = infer_types(data.entities, lambda s, u: completion(s, u, args.base_url, args.model))
        write_json(destination / "entity2label.json", raw)
        write_json(destination / "standardization.json", mapping)
        write_json(destination / "entity2label_clustered_final.json", final)
    elif args.command == "patterns":
        counts, frequencies = type_frequencies(training, data.types, data.relations)
        write_json(destination / "relation_type_counts.json", counts)
        write_json(destination / "relation_type_percentages.json", {r: {p: 100 * f for p, f in pairs.items()} for r, pairs in frequencies.items()})
    else:
        p = read_json(ROOT / "configs/protocol.json")["discovery"]
        miner = RuleMiner(training, data.types, data.relations, data.patterns, p["seed"])
        rules = miner.discover(args.walks or p["walks"], attempts=p["confidence_attempts"], mode=args.variant, epsilon=p["epsilon_type"])
        write_json(destination, rules)
        write_json(destination.with_suffix(".meta.json"), {"dataset": args.dataset, "variant": args.variant,
                   "parameters": dict(p, walks=args.walks or p["walks"]), "rng": "Python random.Random, single worker",
                   "training_canonical_sha256": canonical_hash(training), "rule_sha256": digest(destination),
                   "origin": "newly mined; does not assert identity with supplied historical or external banks"})
    print(f"Wrote new {args.command} artifact: {destination}")


def prepare_command(args):
    meta = prepare(ROOT, args.dataset, args.split, tokenizer_at(args.tokenizer), args.output,
                   args.data_root, args.variant, args.backend, args.max_samples, args.channels, args.operator,
                   not args.repeat_shared_facts, args.omit, args.rules, overwrite=args.overwrite, p_variant=args.p_variant)
    print(json.dumps({k: meta[k] for k in ("dataset", "split", "queries", "scope", "prepared_sha256")}, indent=2))


def train(args):
    import subprocess
    import yaml
    config = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    folder = Path(args.prepared_dir)
    if args.setting != "source-only" and args.dataset is None:
        raise ValueError("Target training requires --dataset")
    datasets = ["GDELT-train", "WIKI-train"] if args.setting == "source-only" else [args.dataset + "-train"]
    info = read_json(folder / "dataset_info.json")
    if any(info.get(name, {}).get("file_name") != name + ".json" for name in datasets):
        raise ValueError("Training dataset_info.json does not reference the requested prepared files")
    metas = [load_prepared(ROOT, folder / (name + ".json"))[1] for name in datasets]
    if len({m["tokenizer_sha256"] for m in metas}) != 1:
        raise ValueError("Training mixtures must use one tokenizer")
    from myst_fusion.workflow import tokenizer_fingerprint
    if tokenizer_fingerprint(tokenizer_at(args.model)) != metas[0]["tokenizer_sha256"]:
        raise ValueError("Training model tokenizer differs from prepared inputs")
    config.update(model_name_or_path=args.model, dataset_dir=str(folder), dataset=",".join(datasets), output_dir=args.output)
    if args.setting == "source-to-target":
        if not args.source_adapter:
            raise ValueError("Source-to-target requires --source-adapter")
        config.update(adapter_name_or_path=args.source_adapter, create_new_adapter=False)
    elif args.source_adapter:
        raise ValueError("Only source-to-target training uses --source-adapter")
    destination = Path(args.config_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Use a fresh generated training config")
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Training config: {destination}")
    if not args.write_only:
        subprocess.run(["llamafactory-cli", "train", str(destination)], check=True)


def predict(args):
    import torch
    from transformers import AutoModelForCausalLM
    from myst_fusion.decoding import SCORING_PROTOCOL, predict_candidates
    tokenizer = tokenizer_at(args.model)
    records, meta = load_prepared(ROOT, args.prepared, tokenizer)
    output = Path(args.output)
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError("Use a fresh prediction output")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {"model": str(Path(args.model).resolve()), "adapter": args.adapter, "prepared_sha256": meta["prepared_sha256"],
                "scoring_protocol": SCORING_PROTOCOL, "tokenizer_sha256": meta["tokenizer_sha256"], "scope": meta["scope"]}
    if args.adapter:
        run_meta["adapter_sha256"] = {p.name: digest(p) for p in Path(args.adapter).glob("adapter_model.*") if p.is_file()}
    write_json(output.with_suffix(".meta.json"), run_meta)
    with output.open("x", encoding="utf-8") as stream:
        for i, record in enumerate(records):
            _, traces = predict_candidates(model, tokenizer, record["input"], meta["entity_vocabulary"],
                                          meta["cutoff_len"], args.second_step_batch_size)
            stream.write(json.dumps({"query_id": record["query_id"], "candidate_sequences": traces,
                         "prepared_sha256": meta["prepared_sha256"], "scoring_protocol": SCORING_PROTOCOL}) + "\n")
            stream.flush()
            if (i + 1) % 100 == 0:
                print(f"Predicted {i + 1}/{len(records)}", flush=True)


def metrics(args):
    records, meta = load_prepared(ROOT, args.prepared)
    predictions = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = replay(records, meta, predictions)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    imp = commands.add_parser("import-data", help="Validate and import externally obtained facts")
    imp.add_argument("--dataset", choices=DATASETS, required=True)
    imp.add_argument("--source", required=True, help="Dataset directory containing train/valid/test.txt")
    imp.add_argument("--format", choices=("processed", "regcn"), default="processed")
    imp.add_argument("--data-root", default="data")
    imp.set_defaults(func=import_data)
    for command in ("types", "patterns", "mine"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=DATASETS, required=True)
        sub.add_argument("--data-root", default="data")
        sub.add_argument("--output", required=True)
        if command == "types":
            sub.add_argument("--base-url", required=True, help="OpenAI-compatible local /v1 endpoint")
            sub.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
        if command == "mine":
            sub.add_argument("--variant", choices=("joint", "uniform", "time", "type"), default="joint")
            sub.add_argument("--walks", type=int)
        sub.set_defaults(func=regenerate)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", choices=DATASETS, required=True)
    prep.add_argument("--split", choices=("train", "valid", "test"), required=True)
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--data-root", default="data")
    prep.add_argument("--variant", choices=("joint", "uniform", "time", "type"), default="joint")
    prep.add_argument("--rules", help="New rule bank from the mine command")
    prep.add_argument("--backend", choices=("python", "native"), default="python")
    prep.add_argument("--max-samples", type=int)
    prep.add_argument("--channels", choices=("T", "P", "L", "TP", "TL", "PL", "TPL"), default="TPL")
    prep.add_argument("--operator", choices=("weighted", "max", "mul"), default="weighted")
    prep.add_argument("--p-variant", choices=("bounded", "type_frequency", "recurrence", "product"), default="bounded")
    prep.add_argument("--repeat-shared-facts", action="store_true")
    prep.add_argument("--omit", nargs="*", choices=("entity", "relation", "type"), default=[])
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=prepare_command)
    tr = commands.add_parser("train")
    tr.add_argument("--setting", choices=("source-only", "target-only", "source-to-target"), required=True)
    tr.add_argument("--dataset", choices=TARGETS)
    tr.add_argument("--model", required=True)
    tr.add_argument("--source-adapter")
    tr.add_argument("--prepared-dir", default="processed")
    tr.add_argument("--output", required=True)
    tr.add_argument("--config-output", required=True)
    tr.add_argument("--write-only", action="store_true")
    tr.set_defaults(func=train)
    pr = commands.add_parser("predict")
    pr.add_argument("--prepared", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--adapter")
    pr.add_argument("--second-step-batch-size", type=int, default=8)
    pr.add_argument("--output", required=True)
    pr.set_defaults(func=predict)
    met = commands.add_parser("metrics")
    met.add_argument("--prepared", required=True)
    met.add_argument("--predictions", required=True)
    met.add_argument("--output", required=True)
    met.set_defaults(func=metrics)
    return p


if __name__ == "__main__":
    os.chdir(ROOT)
    args = parser().parse_args()
    args.func(args)
