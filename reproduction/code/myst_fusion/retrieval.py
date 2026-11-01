"""Complete directed witnesses and response Equations 12--17."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from .data import validate_rules
from .mining import augmented, variables_match


@dataclass(frozen=True)
class Chain:
    endpoint: int
    facts: tuple
    channels: tuple
    score: float


def normalize(values):
    maximum = max(values.values(), default=0.0)
    return {e: v / maximum if maximum else 0.0 for e, v in values.items()}


def semantic_scores(frequencies, recurrence, beta, variant="bounded"):
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0,1]")
    fh, nh = normalize(frequencies), normalize(recurrence)
    if variant == "type_frequency":
        return fh
    if variant == "recurrence":
        return nh
    if variant == "product":
        return normalize({e: fh[e] * nh[e] for e in frequencies})
    if variant != "bounded":
        raise ValueError("Unknown semantic-score definition")
    return normalize({e: nh[e] * (1 + beta * fh[e]) for e in frequencies})


class Retriever:
    def __init__(self, facts, rules, types, relations, patterns, backend="python"):
        self.facts = sorted(facts)  # Preserve observed occurrences for recurrence.
        self.rules, self.types, self.relations, self.patterns = rules, types, relations, patterns
        self.nr, self.backend = len(relations), backend
        validate_rules(rules, self.nr)
        self.outgoing = None
        self.native = None
        if backend == "native":
            import grounded_chain_native
            self.native = grounded_chain_native
        elif backend != "python":
            raise ValueError("backend must be python or native")

    def python_pool(self, known, relation, direction, time):
        if self.outgoing is None:
            self.outgoing = defaultdict(list)
            for f in sorted(set(augmented(self.facts, self.nr))):
                self.outgoing[(f[0], f[1])].append(f)
        past = [f for f in self.facts if f[3] < time]
        if not past:
            return {}, Counter()
        scale = time - min(f[3] for f in past)
        recurrence = Counter((o if direction == 0 else h) for h, _, o, _ in past
                             if (h == known if direction == 0 else o == known))
        pool = {}
        def record(path, logical):
            path = tuple(path)
            temporal = sum(math.exp(-(time - f[3]) / scale) for f in path) / len(path)
            old = pool.get(path)
            pool[path] = (temporal, max(logical, old[1] if old else 0.0))
        for h, r, o, t in set(past):
            if direction == 0 and h == known:
                record(((h, r, o, t),), 0.0)
            elif direction == 1 and o == known:
                record(((o, r + self.nr, h, t),), 0.0)
        for rule in self.rules.get(str(relation + direction * self.nr), []):
            def visit(node, path, nodes):
                if len(path) == len(rule["body_rels"]):
                    record(path, rule["conf"])
                    return
                r = rule["body_rels"][len(path)]
                for fact in self.outgoing[(node, r)]:
                    if fact[3] >= time or (path and fact[3] < path[-1][3]):
                        continue
                    extended = nodes + [fact[2]]
                    if variables_match(extended, rule.get("var_constraints", [])):
                        visit(fact[2], path + [fact], extended)
            visit(known, [], [known])
        return pool, recurrence

    def native_pool(self, known, relation, direction, time):
        # Backend globals belong to the extension, so reinitialize explicitly.
        if getattr(self.native, "_owner", None) is not self:
            self.native.init(self.facts, self.rules, self.nr)
            self.native._owner = self
        front = self.native.query(known, relation, direction, time)
        pool = {}
        for paths in front.values():
            for temporal, logical, witness in paths:
                directed = []
                for fid, r in witness:
                    h, _, o, t = self.facts[fid]
                    directed.append((h, r, o, t) if r < self.nr else (o, r, h, t))
                pool[tuple(directed)] = (temporal, logical)
        return pool, self.native.recurrence(known, direction, time)

    def retrieve(self, known, relation, direction, time, alpha, logical_weight, beta,
                 budget=50, channels="TPL", operator="weighted", p_variant="bounded"):
        if (known < 0 or relation not in self.relations or direction not in (0, 1) or budget < 0
                or not channels or channels not in ("T", "P", "L", "TP", "TL", "PL", "TPL")
                or any(not math.isfinite(w) or w <= 0 for w in (alpha, logical_weight))):
            raise ValueError("Invalid retrieval parameters")
        if operator not in ("weighted", "max", "mul"):
            raise ValueError("Invalid fusion operator")
        # Positive additive TPL is exactly optimized by the native T/L frontier.
        # Other operators/channels enumerate all witnesses to preserve zero/tie cases.
        pool_fn = self.native_pool if self.native and channels == "TPL" and operator == "weighted" else self.python_pool
        pool, recurrence = pool_fn(known, relation, direction, time)
        if not pool:
            return []
        endpoints = {p[-1][2] for p in pool}
        frequencies = {}
        for e in endpoints:
            a, b = (known, e) if direction == 0 else (e, known)
            pair = f"{self.types.get(a)}->{self.types.get(b)}"
            frequencies[e] = self.patterns.get(self.relations[relation], {}).get(pair, 0.0)
        if any(not math.isfinite(v) or v < 0 for v in frequencies.values()):
            raise ValueError("Invalid type frequency")
        semantic = semantic_scores(frequencies, {e: recurrence.get(e, 0) for e in endpoints}, beta, p_variant)
        max_t = max(t for t, _ in pool.values())
        max_l = max(l for _, l in pool.values())
        best = {}
        for path, (temporal, logical) in pool.items():
            e = path[-1][2]
            values = (temporal / max_t, semantic[e], logical / max_l if max_l else 0.0)
            active = [v for c, v in zip("TPL", values) if c in channels]
            if operator == "weighted":
                score = sum(w * v for c, w, v in zip("TPL", (1, alpha, logical_weight), values) if c in channels)
            elif operator == "max":
                score = max(active)
            else:
                score = math.prod(active)
            key = (-score, len(path), -path[-1][3], path)
            if e not in best or key < best[e][0]:
                best[e] = (key, Chain(e, path, values, score))
        return sorted((row[1] for row in best.values()), key=lambda c: (-c.score, c.endpoint))[:budget]
