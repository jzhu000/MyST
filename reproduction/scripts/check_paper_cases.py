"""Recompute the three Table 10 retrieval witnesses; no predictor or GPU needed."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from myst_fusion.data import Dataset, read_json
from myst_fusion.retrieval import Retriever
from myst_fusion.workflow import check_benchmark

# Entity/relation names, times, ranks, and normalized scores in response Table 10.
CASES = [
    ("ICEWS14", "Business (Taiwan)", "Make an appeal or request", "Kuomintang", 343, 1, 28,
     (0.7453, 0.0, 0.4774), 1.2227, [26, 272, 272]),
    ("ICEWS18", "Party Member (Thailand)", "Host a visit", "Hong Kong", 281, 1, 11,
     (0.6015, 0.0, 0.5178), 1.1194, [88, 179]),
    ("YAGO", "Pavel Plaskonny", "playsFor", "FC Belshina Bobruisk", 185, 0, 15,
     (0.9964, 0.0, 0.2650), 1.2614, [182, 184, 184]),
]


def check(data_root, backend):
    config = read_json(ROOT / "configs/protocol.json")
    results = []
    for dataset, known, relation, answer, time, direction, expected_rank, expected_channels, expected_score, times in CASES:
        data = Dataset(ROOT, dataset, data_root).load_parts()
        for split, facts in data.parts.items():
            check_benchmark(ROOT, dataset, split, facts)
        entities = {name: i for i, name in data.entities.items()}
        relations = {name: i for i, name in data.relations.items()}
        known_id, answer_id = entities[known], entities[answer]
        p = config["datasets"][dataset]
        history = data.history("test")
        if any(t < time and {h, o} == {known_id, answer_id} for h, _, o, t in history):
            raise AssertionError(f"{dataset}: expected no past direct fact between the query pair")
        chains = Retriever(history, data.rules, data.types, data.relations, data.patterns, backend).retrieve(
            known_id, relations[relation], direction, time, p["alpha"], p["lambda"], p["beta"], len(entities))
        matches = [(rank, c) for rank, c in enumerate(chains, 1) if c.endpoint == answer_id]
        if not matches or matches[0][0] != expected_rank:
            raise AssertionError(f"{dataset}: expected answer rank {expected_rank}; got {matches[0][0] if matches else None}")
        rank, witness = matches[0]
        if [f[3] for f in witness.facts] != times or len(witness.facts) != len(times):
            raise AssertionError(f"{dataset}: representative body differs from Table 10")
        if any(abs(a - b) > 0.000051 for a, b in zip(witness.channels, expected_channels)):
            raise AssertionError(f"{dataset}: normalized channels differ beyond four-decimal rounding")
        if abs(witness.score - expected_score) > 0.000051:
            raise AssertionError(f"{dataset}: fused score differs beyond four-decimal rounding")
        results.append({"dataset": dataset, "rank": rank, "channels": witness.channels,
                        "score": witness.score, "facts": witness.facts, "status": "passed"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    args = parser.parse_args()
    print(json.dumps(check(Path(args.data_root).resolve(), args.backend), indent=2))
