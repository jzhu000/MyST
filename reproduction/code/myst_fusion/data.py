"""Fixed benchmark IDs, visible history, and label-side time-aware filters."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import random

DATASETS = ("ICEWS14", "ICEWS18", "YAGO", "GDELT", "WIKI")
TARGETS = DATASETS[:3]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path, lf=False):
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read_mapping(path):
    """Return integer ID -> name, rejecting ambiguous mappings."""
    pairs = [line.rsplit("\t", 1) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    result = {int(i): name for name, i in pairs}
    if len(result) != len(pairs) or len(set(result.values())) != len(pairs):
        raise ValueError(f"Duplicate ID or name: {path}")
    if not result or min(result) < 0:
        raise ValueError(f"Expected nonnegative benchmark IDs: {path}")
    return result


def read_facts(path):
    facts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = tuple(map(int, line.split()))
        if len(row) not in (4, 5):
            raise ValueError(f"Expected four fact columns and optional legacy fifth column: {path}")
        facts.append(row[:4])
    return facts


def validate_facts(facts, entities, relations):
    if any(h not in entities or o not in entities or r not in relations or t < 0 for h, r, o, t in facts):
        raise ValueError("Facts reference an unknown ID or negative timestamp")


def apply_entity_redirects(facts, redirects):
    """Apply a declared upstream-to-fixed ID map, preserving every row and time."""
    return [(redirects.get(h, h), r, redirects.get(o, o), t) for h, r, o, t in facts]


def validate_rules(rules, nr):
    for head, bank in rules.items():
        if not 0 <= int(head) < 2 * nr:
            raise ValueError("Invalid rule head ID")
        for rule in bank:
            body, groups = rule["body_rels"], rule.get("var_constraints", [])
            if int(rule["head_rel"]) != int(head) or not 1 <= len(body) <= 3:
                raise ValueError("Invalid rule head/body")
            if any(not isinstance(r, int) or not 0 <= r < 2 * nr for r in body):
                raise ValueError("Invalid directed body predicate")
            flat = [v for group in groups for v in group]
            if (any(len(group) < 2 for group in groups) or len(flat) != len(set(flat))
                    or any(not isinstance(v, int) or not 0 <= v <= len(body) for v in flat)):
                raise ValueError("Invalid repeated-variable constraints")
            if not 0 <= rule["conf"] <= 1:
                raise ValueError("Invalid rule confidence")
            if "body_supp" in rule and not 0 <= rule["rule_supp"] <= rule["body_supp"]:
                raise ValueError("Invalid rule support")


class Dataset:
    def __init__(self, root, name, data_root="data", variant="joint", rules_path=None):
        self.root, self.name = Path(root), name
        self.asset = self.root / "assets" / name
        self.entities = read_mapping(self.asset / "entity2id.txt")
        self.relations = read_mapping(self.asset / "relation2id.txt")
        if set(self.relations) != set(range(len(self.relations))):
            raise ValueError("Relation IDs must be contiguous for inverse offsets")
        redirect_path = self.asset / "entity_id_redirects.json"
        self.entity_redirects = {int(a): int(b) for a, b in read_json(redirect_path).items()} if redirect_path.exists() else {}
        if any(a in self.entities or b not in self.entities for a, b in self.entity_redirects.items()):
            raise ValueError("Declared redirects must map absent upstream aliases to fixed entity IDs")
        labels = read_json(self.asset / "entity2label_clustered_final.json")
        if set(labels) != set(self.entities.values()):
            raise ValueError("Final type labels must cover the fixed entity vocabulary exactly")
        self.types = {i: labels[n] for i, n in self.entities.items()}
        # Archived files use percentages; Equation 11 uses fractions.
        percentages = read_json(self.asset / "relation_type_percentages.json")
        self.patterns = {r: {p: f / 100.0 for p, f in pairs.items()} for r, pairs in percentages.items()}
        self.rule_path = Path(rules_path) if rules_path else self.asset / "rules" / (variant + ".json")
        self.rules = read_json(self.rule_path)
        validate_rules(self.rules, len(self.relations))
        self.parts = {}
        self.data_dir = Path(data_root) / name

    def load_parts(self):
        for part in ("train", "valid", "test"):
            self.parts[part] = read_facts(self.data_dir / (part + ".txt"))
            validate_facts(self.parts[part], self.entities, self.relations)
        return self

    def history(self, split):
        visible = {"train": ("train",), "valid": ("train", "valid"), "test": ("train", "valid", "test")}[split]
        return [f for p in visible for f in self.parts[p]]

    def queries(self, split, max_samples=None, seed=42):
        # Labels are isolated from the retrieval path; only metrics inspect them.
        facts = self.parts[split]
        indices = range(2 * len(facts))
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("Sample size must be positive")
            indices = random.Random(seed).sample(indices, min(max_samples, len(indices)))
        queries = []
        wanted = set()
        for index in indices:
            row, direction = divmod(index, 2)
            h, r, o, t = facts[row]
            known, answer = (h, o) if direction == 0 else (o, h)
            wanted.add((known, r, t, direction))
            queries.append({"query_id": f"{self.name}:{split}:{row}:{direction}", "known": known,
                            "relation": r, "time": t, "direction": direction, "answer": str(answer)})
        positives = defaultdict(set)
        for facts in self.parts.values():
            for h, r, o, t in facts:
                if (h, r, t, 0) in wanted:
                    positives[(h, r, t, 0)].add(str(o))
                if (o, r, t, 1) in wanted:
                    positives[(o, r, t, 1)].add(str(h))
        for q in queries:
            q["filters"] = sorted(positives[(q["known"], q["relation"], q["time"], q["direction"])] - {q["answer"]}, key=int)
        return sorted(queries, key=lambda q: (q["time"], q["query_id"]))
