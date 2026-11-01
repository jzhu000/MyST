"""Appendix A typing/standardization and training-only relation type frequencies."""
from collections import Counter, defaultdict
import json
import os
from urllib.request import Request, urlopen

TYPE_SYSTEM = "You are an entity type classifier. Determine the entity type of the given word and only return the type name.\nThe first character of the entity type name must be capitalized."
CLUSTER_SYSTEM = """You are an expert in entity type clustering. Your task is to group semantically similar entity type labels together and assign a standardized label name to each cluster.

Please follow these rules:
1. Group labels that are semantically identical or very similar (e.g., "Person", "PERSON", "person" should be grouped together)
2. Choose a standardized name for each cluster, using title case (e.g., "Person" instead of "PERSON" or "person")
3. Do not include any explanatory text in the JSON, only the mapping from original labels to standardized labels
4. Ensure the returned JSON is valid, with both keys and values enclosed in double quotes
5. Do not lose any labels; ensure the number of output labels matches the input, and the output labels are consistent with the original labelsFor example, for the label list ["person", "PERSON", "Organization", "ORG"], you should return:
{"person": "Person", "PERSON": "Person", "Organization": "Organization", "ORG": "Organization"}"""


def completion(system, user, base_url, model="meta-llama/Meta-Llama-3-8B"):
    payload = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}]}
    request = Request(base_url.rstrip("/") + "/chat/completions", json.dumps(payload).encode(),
                      {"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("MYST_TYPING_API_KEY", "EMPTY")})
    with urlopen(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()


def infer_types(entities, call):
    raw = {name: call(TYPE_SYSTEM, f"Determine the entity type of '{name}'.") for name in entities.values()}
    if any(not label.strip() for label in raw.values()):
        raise ValueError("Empty generated type")
    labels = sorted(set(raw.values()))
    mapping = json.loads(call(CLUSTER_SYSTEM,
        "Please cluster the following entity type labels, grouping semantically identical or very similar labels together, and assign a standardized label name to each cluster.\n"
        "Return a JSON-formatted mapping where the keys are the original labels and the values are the standardized label names.\n\n"
        "Label list:\n" + json.dumps(labels) + "\n\nOnly return the JSON-formatted mapping result, without any additional explanations. Ensure the returned JSON is valid, with both keys and values enclosed in double quotes."))
    if set(mapping) != set(labels) or any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
        raise ValueError("Standardization must map every dataset-local raw label to one nonempty type")
    return raw, mapping, {name: mapping[label] for name, label in raw.items()}


def type_frequencies(training_facts, types, relations):
    counts, totals = defaultdict(Counter), Counter()
    for h, r, o, _ in training_facts:
        name = relations[r]
        totals[name] += 1
        pair = f"{types.get(h, 'UNKNOWN')}->{types.get(o, 'UNKNOWN')}"
        counts[name][pair] += 1
    frequencies = {r: {pair: n / totals[r] for pair, n in pairs.items()} for r, pairs in counts.items()}
    return dict(counts), frequencies
