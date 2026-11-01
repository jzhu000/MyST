"""Preparation and replayable evaluation, without training-framework imports."""
from pathlib import Path
import hashlib
import json
import math

from . import PROTOCOL
from .data import Dataset, digest, read_json, write_json
from .decoding import SCORING_PROTOCOL, rank_record, merge_query_records
from .prompt import pack, encode
from .retrieval import Retriever


def tokenizer_fingerprint(tokenizer):
    value = {"vocab": tokenizer.get_vocab(), "specials": tokenizer.special_tokens_map,
             "class": type(tokenizer).__name__}
    if hasattr(tokenizer, "backend_tokenizer"):
        value["backend"] = tokenizer.backend_tokenizer.to_str()
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def canonical_hash(facts):
    return hashlib.sha256("".join("\t".join(map(str, f)) + "\n" for f in facts).encode()).hexdigest()


def check_benchmark(root, dataset, split, facts):
    expected = read_json(Path(root) / "configs/benchmark_checksums.json")[dataset][split]
    if len(facts) != expected["rows"] or canonical_hash(facts) != expected["canonical_sha256"]:
        raise ValueError(f"Benchmark facts/order differ: {dataset}/{split}; refusing to change IDs or reuse weights silently")


def prepare(root, dataset, split, tokenizer, output, data_root="data", variant="joint", backend="python",
            max_samples=None, channels="TPL", operator="weighted", deduplicate=True, omit=(), rules_path=None,
            check_data=True, overwrite=False, p_variant="bounded"):
    root, output = Path(root), Path(output)
    meta_path = output.with_suffix(".meta.json")
    if (output.exists() or meta_path.exists()) and not overwrite:
        raise FileExistsError(f"Prepared output exists: {output}; use a new name or --overwrite")
    data = Dataset(root, dataset, data_root, variant, rules_path).load_parts()
    if check_data:
        for part, facts in data.parts.items():
            check_benchmark(root, dataset, part, facts)
    config = read_json(root / "configs/protocol.json")
    params = config["datasets"][dataset]
    retriever = Retriever(data.history(split), data.rules, data.types, data.relations, data.patterns, backend)
    full_count = 2 * len(data.parts[split])
    if max_samples is None and split == "train":
        max_samples = params.get("source_examples")
    queries = data.queries(split, max_samples, config["sampling_seed"])
    reserve = max(len(encode(tokenizer, str(e))) for e in data.entities) + 1
    if reserve > 3:
        raise ValueError("Some mapped GIDs exceed the two-token decoder; use the specified tokenizer")
    files = list(data.asset.glob("*.json")) + list(data.asset.glob("*.txt")) + [data.rule_path]
    files += [data.data_dir / (part + ".txt") for part in ("train", "valid", "test")]
    files += [root / "configs/protocol.json", root / "configs/benchmark_checksums.json"]
    files += sorted((root / "code/myst_fusion").glob("*.py"))
    if backend == "native":
        files.append(root / "code/native/grounded_chain_retriever.cpp")
    fingerprints = {}
    for path in files:
        resolved = path.resolve()
        key = resolved.relative_to(root.resolve()).as_posix() if resolved.is_relative_to(root.resolve()) else str(resolved)
        fingerprints[key] = digest(path)
    meta = {"protocol": PROTOCOL, "dataset": dataset, "split": split,
            "entity_vocabulary": {str(e): str(e) for e in data.entities},
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer), "input_sha256": fingerprints,
            "cutoff_len": config["cutoff_len"], "parameters": params, "variant": variant,
            "channels": channels, "operator": operator, "p_variant": p_variant, "deduplicate": deduplicate, "omit": list(omit),
            "backend": backend, "queries": len(queries), "full_queries": full_count,
            "scope": "complete" if len(queries) == full_count else "sample", "benchmark_verified": check_data}
    records = []
    for i, q in enumerate(queries):
        chains = retriever.retrieve(q["known"], q["relation"], q["direction"], q["time"],
                                    params["alpha"], params["lambda"], params["beta"], config["endpoint_budget"], channels, operator, p_variant)
        # Only query fields and permitted evidence enter packing; never the answer.
        prompt, packing = pack(q["known"], q["relation"], q["direction"], q["time"], chains,
                               data.entities, data.relations, data.types, tokenizer, config["cutoff_len"], deduplicate, omit, reserve)
        records.append({"instruction": "", "input": prompt, "output": q["answer"],
                        "filters": q["filters"], "query_id": q["query_id"], "packing": packing})
        if (i + 1) % 1000 == 0:
            print(f"{dataset}/{split}: {i + 1}/{len(queries)}", flush=True)
    if not records:
        raise ValueError("No queries")
    write_json(output, records)
    meta["prepared_sha256"] = digest(output)
    write_json(meta_path, meta)
    info_path = output.parent / "dataset_info.json"
    info = read_json(info_path) if info_path.exists() else {}
    info[output.stem] = {"file_name": output.name, "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    write_json(info_path, info)
    return meta


def load_prepared(root, path, tokenizer=None):
    path, root = Path(path), Path(root)
    meta = read_json(path.with_suffix(".meta.json"))
    if meta["protocol"] != PROTOCOL or digest(path) != meta["prepared_sha256"]:
        raise ValueError("Prepared protocol/content mismatch")
    for name, expected in meta["input_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")
    if tokenizer is not None and tokenizer_fingerprint(tokenizer) != meta["tokenizer_sha256"]:
        raise ValueError("Prediction tokenizer differs from preparation")
    records = read_json(path)
    ids = [r["query_id"] for r in records]
    if len(ids) != meta["queries"] or len(set(ids)) != len(ids) or not ids:
        raise ValueError("Missing/duplicate prepared queries")
    if any(r["output"] not in meta["entity_vocabulary"] for r in records):
        raise ValueError("Invalid gold GID")
    return records, meta


def consolidate(traces, vocabulary):
    best = {}
    for trace in traces:
        gid, tokens, score = trace["gid"], trace["token_ids"], trace["log_joint_probability"]
        if gid not in vocabulary or len(tokens) != 2 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid complete-GID score trace")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score > 0:
            raise ValueError("Invalid log joint probability")
        if gid not in best or (-score, tuple(tokens)) < (-best[gid]["log_joint_probability"], tuple(best[gid]["token_ids"])):
            best[gid] = trace
    return list(best.values())


def replay(records, meta, predictions):
    expected = {r["query_id"] for r in records}
    by_id = {}
    for p in predictions:
        if p.get("scoring_protocol") != SCORING_PROTOCOL or p.get("prepared_sha256") != meta["prepared_sha256"]:
            raise ValueError("Prediction protocol or prepared fingerprint mismatch")
        if p["query_id"] not in expected or p["query_id"] in by_id:
            raise ValueError("Extra or duplicate prediction query")
        by_id[p["query_id"]] = p
    ranked = []
    for record in records:
        traces = consolidate(by_id.get(record["query_id"], {}).get("candidate_sequences", []), meta["entity_vocabulary"])
        ranked.append(rank_record(record["query_id"], [r["gid"] for r in traces], meta["entity_vocabulary"],
                                  record["output"], record["filters"], traces))
    metrics, ranked = merge_query_records(ranked, [r["query_id"] for r in records])
    return {"metrics": metrics, "queries": len(records), "missing_predictions": len(expected - set(by_id)),
            "scope": meta["scope"], "full_queries": meta["full_queries"], "records": ranked}
