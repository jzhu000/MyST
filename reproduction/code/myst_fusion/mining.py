"""Head-seeded backward discovery and independent uniform body confidence.

Uses a local Python RNG. Remined banks are new artifacts, never byte-for-byte
reproductions of historical NumPy/multi-worker rule banks.
"""
from collections import defaultdict
import math
import random


def variable_groups(nodes):
    groups = defaultdict(list)
    for i, e in enumerate(nodes):
        groups[e].append(i)
    return sorted(g for g in groups.values() if len(g) > 1)


def variables_match(nodes, groups):
    # Distinct variables need not be distinct entities. Only stated equalities bind.
    return all(len({nodes[i] for i in group if i < len(nodes)}) <= 1 for group in groups)


def augmented(facts, nr):
    return list(facts) + [(o, r + nr, h, t) for h, r, o, t in facts]


def transition_probabilities(edges, types, relations, patterns, mode="joint", epsilon=0.01):
    if not edges:
        return []
    if mode not in ("joint", "uniform", "time", "type") or not 0 < epsilon < float("inf"):
        raise ValueError("Invalid discovery mode or missing-pair weight")
    nr, latest = len(relations), max(f[3] for f in edges)
    weights = []
    for h, r, o, t in edges:
        if r >= nr:
            h, o = o, h
        pair = f"{types.get(h)}->{types.get(o)}"
        value = patterns.get(relations[r % nr], {}).get(pair, 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid training type frequency")
        type_weight = value if value > 0 else epsilon
        # Subtracting a common timestamp cancels in Equation 7 and prevents underflow.
        weights.append((math.exp(t - latest) if mode in ("joint", "time") else 1.0)
                       * (type_weight if mode in ("joint", "type") else 1.0))
    total = sum(weights)
    return [w / total for w in weights]


class RuleMiner:
    def __init__(self, training_facts, types, relations, patterns, seed=123):
        self.nr, self.types, self.relations, self.patterns = len(relations), types, relations, patterns
        self.rng = random.Random(seed)
        self.edges, self.neighbors, self.head_times = defaultdict(list), defaultdict(list), defaultdict(list)
        for f in augmented(training_facts, self.nr):
            h, r, o, t = f
            self.edges[r].append(f)
            self.neighbors[(h, r)].append(f)
            self.head_times[(h, r, o)].append(t)
        self.outgoing = defaultdict(list)
        for (node, _), edges in self.neighbors.items():
            self.outgoing[node].extend(edges)

    def confidence(self, rule, attempts=500):
        bodies = set()
        first_edges = self.edges[rule["body_rels"][0]]
        if not first_edges:
            return 0.0, 0, 0
        for _ in range(attempts):
            first = self.rng.choice(first_edges)
            path, nodes = [first], [first[0], first[2]]
            for rel in rule["body_rels"][1:]:
                legal = [e for e in self.neighbors[(nodes[-1], rel)] if e[3] >= path[-1][3]]
                if not legal:
                    break
                edge = self.rng.choice(legal)
                path.append(edge)
                nodes.append(edge[2])
            if len(path) == len(rule["body_rels"]) and variables_match(nodes, rule["var_constraints"]):
                bodies.add(tuple(path))
        supported = sum(any(t > body[-1][3] for t in self.head_times[(body[0][0], rule["head_rel"], body[-1][2])]) for body in bodies)
        return supported / len(bodies) if bodies else 0.0, supported, len(bodies)

    def discover(self, walks=200, lengths=(1, 2, 3), attempts=500, mode="joint", epsilon=0.01):
        if walks < 1 or attempts < 1 or any(n not in (1, 2, 3) for n in lengths):
            raise ValueError("Invalid mining budget or body length")
        rules, seen = defaultdict(list), set()
        for head_rel in sorted(self.edges):
            for length in lengths:
                for _ in range(walks):
                    head = self.rng.choice(self.edges[head_rel])
                    node, time, backward, previous = head[2], head[3], [], head
                    for step in range(length):
                        inverse = (previous[2], (previous[1] + self.nr) % (2 * self.nr), previous[0], previous[3])
                        legal = [e for e in self.outgoing[node]
                                 if (e[3] < time if step == 0 else e[3] <= time)
                                 and e != inverse and (step != length - 1 or e[2] == head[0])]
                        if not legal:
                            break
                        edge = self.rng.choices(legal, weights=transition_probabilities(
                            legal, self.types, self.relations, self.patterns, mode, epsilon), k=1)[0]
                        backward.append(edge)
                        node, time, previous = edge[2], edge[3], edge
                    if len(backward) != length:
                        continue
                    body = [(o, (r + self.nr) % (2 * self.nr), h, t) for h, r, o, t in reversed(backward)]
                    groups = variable_groups([body[0][0]] + [f[2] for f in body])
                    key = head_rel, tuple(f[1] for f in body), tuple(map(tuple, groups))
                    if key in seen:
                        continue
                    seen.add(key)
                    rule = {"head_rel": head_rel, "body_rels": list(key[1]), "var_constraints": groups}
                    conf, supp, total = self.confidence(rule, attempts)
                    if conf:
                        rules[str(head_rel)].append(dict(rule, conf=conf, rule_supp=supp, body_supp=total))
        for bank in rules.values():
            bank.sort(key=lambda r: (-r["conf"], r["body_rels"], r["var_constraints"]))
        return dict(rules)
