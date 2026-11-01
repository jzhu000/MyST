"""Seven-section fixed-GID prompt with whole-chain context packing."""


def encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def serialize(known, relation, direction, time, chains, entities, relations, types, deduplicate=True,
              omit=()):
    nr = len(relations)
    query_relation = relation + direction * nr
    history, memberships, first = [], [], {}
    for chain in chains:
        rows = []
        for fact in chain.facts:
            if not deduplicate or fact not in first:
                first[fact] = len(history)
                history.append(fact)
            rows.append(first[fact])
        memberships.append((chain.endpoint, rows))
    ids = sorted({known} | {e for h, _, o, _ in history for e in (h, o)})
    rels = sorted({query_relation} | {r for _, r, _, _ in history})
    def rel_name(r):
        return relations[r] if r < nr else relations[r - nr] + "^-1"
    sections = []
    if "entity" not in omit:
        sections.append(("Entities", [f"{e}:{entities[e]}" for e in ids]))
    if "relation" not in omit:
        sections.append(("Relations", [f"{r}:{rel_name(r)}" for r in rels]))
    if "type" not in omit:
        sections.append(("Entity Types", [f"{e}:{types[e]}" for e in ids if e in types]))
    sections.extend([
        ("History", [f"{t}:[{h},{r},{o}]" for h, r, o, t in history]),
        ("Complete Chains", [f"{e}:[{','.join(map(str, rows))}]" for e, rows in memberships]),
        ("Query", [f"{time}:[{known},{query_relation},?]"]), ("Answer", [])])
    return "\n\n".join("### " + title + " ###\n" + "\n".join(lines) for title, lines in sections) + "\n"


def pack(known, relation, direction, time, chains, entities, relations, types, tokenizer,
         cutoff=1536, deduplicate=True, omit=(), answer_reserve=None):
    # Reserve enough space for every mapped answer, independently of the gold label.
    reserve = answer_reserve if answer_reserve is not None else max(len(encode(tokenizer, str(e))) for e in entities) + 1
    retained = list(chains)
    while True:
        prompt = serialize(known, relation, direction, time, retained, entities, relations, types, deduplicate, omit)
        size = len(encode(tokenizer, prompt))
        if size + reserve <= cutoff:
            return prompt, {"selected_endpoints": [c.endpoint for c in chains],
                            "serialized_endpoints": [c.endpoint for c in retained],
                            "prompt_tokens": size, "reserved_answer_tokens": reserve,
                            "chains": [{"endpoint": c.endpoint, "facts": c.facts,
                                        "channels": c.channels, "score": c.score} for c in retained]}
        if not retained:
            raise ValueError("Base query and mappings exceed the context budget")
        retained.pop()
