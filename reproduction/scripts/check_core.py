"""Small deterministic protocol checks; no model download, GPU, or data required."""
from pathlib import Path
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from myst_fusion.data import Dataset, write_json, apply_entity_redirects
from myst_fusion.decoding import decode_sequence, observe_pairs, rank_record, SCORING_PROTOCOL
from myst_fusion.entity_types import infer_types, type_frequencies
from myst_fusion.mining import RuleMiner, transition_probabilities, variables_match
from myst_fusion.prompt import pack, serialize
from myst_fusion.retrieval import Chain, Retriever, semantic_scores
from myst_fusion.workflow import canonical_hash, consolidate, prepare, load_prepared, replay


class Tokenizer:
    pad_token_id, eos_token_id = 0, 0
    special_tokens_map = {"eos_token": "<eos>"}
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def get_vocab(self):
        return {str(i): i for i in range(256)}
    def decode(self, ids, skip_special_tokens=False):
        pieces = {0: "<eos>", 1: "1", 2: "23", 3: "123", 4: "x", 5: "0", 6: "12", 7: "junk", 10: "100"}
        return "".join(pieces.get(i, f"z{i}") for i in ids)


def rule(body, conf=0.8, head=0, groups=None):
    return {"head_rel": head, "body_rels": body, "var_constraints": groups or [], "conf": conf}


class CoreChecks(unittest.TestCase):
    def test_declared_wiki_alias_conversion_preserves_rows_and_is_idempotent(self):
        facts = [(7, 1, 3, 4), (3, 0, 7, 5), (3, 0, 7, 5)]
        converted = apply_entity_redirects(facts, {7: 2})
        self.assertEqual(converted, [(2, 1, 3, 4), (3, 0, 2, 5), (3, 0, 2, 5)])
        self.assertEqual(len(converted), len(facts))
        self.assertEqual(apply_entity_redirects(converted, {7: 2}), converted)

    def test_types_are_dataset_local_and_all_labels_must_be_mapped(self):
        calls = []
        def call(system, user):
            calls.append(user)
            if "Label list:" in user:
                return json.dumps({"person": "Person", "PERSON": "Person"})
            return "person" if "Alice" in user else "PERSON"
        raw, mapping, final = infer_types({0: "Alice", 1: "Bob"}, call)
        self.assertEqual(set(final.values()), {"Person"})
        self.assertEqual(len(raw), 2)
        self.assertEqual(len(calls), 3)
        with self.assertRaises(ValueError):
            infer_types({0: "Alice"}, lambda s, u: "{}" if "Label list:" in u else "person")

    def test_training_frequency_is_conditioned_on_relation(self):
        counts, frequencies = type_frequencies([(0, 0, 1, 1), (0, 0, 2, 2), (2, 1, 1, 3)],
                                               {0: "A", 1: "B", 2: "C"}, {0: "r", 1: "s"})
        self.assertEqual(frequencies["r"], {"A->B": 0.5, "A->C": 0.5})
        self.assertEqual(counts["s"], {"C->B": 1})

    def test_transition_product_is_renormalized_and_inverse_uses_base_order(self):
        edges = [(1, 1, 0, 2), (2, 1, 0, 2)]
        probabilities = transition_probabilities(edges, {0: "A", 1: "B", 2: "C"}, {0: "r"},
                                                 {"r": {"A->B": 0.2, "A->C": 0.8}})
        self.assertAlmostEqual(probabilities[0], 0.2)
        self.assertAlmostEqual(sum(probabilities), 1)
        missing = transition_probabilities([(0, 0, 1, 99999), (0, 0, 2, 99999)],
                                           {0: "A", 1: "B", 2: "C"}, {0: "r"}, {"r": {"A->B": 0.5}})
        self.assertAlmostEqual(missing[1], 0.01 / 0.51)
        self.assertEqual(transition_probabilities([], {}, {}, {}), [])

    def test_body_confidence_keeps_unsupported_bodies_and_counts_later_heads_once(self):
        facts = [(0, 1, 1, 1), (2, 1, 3, 1), (0, 0, 1, 2), (0, 0, 1, 3)]
        miner = RuleMiner(facts, {}, {0: "r", 1: "s"}, {})
        conf, supp, total = miner.confidence(rule([1]), 500)
        self.assertEqual((conf, supp, total), (0.5, 1, 2))
        self.assertEqual(miner.confidence(rule([1, 0]), 10), (0, 0, 0))
        self.assertTrue(variables_match([0, 1, 0], [[0, 2]]))
        self.assertFalse(variables_match([0, 1, 2], [[0, 2]]))
        self.assertTrue(variables_match([0, 0, 0], []))

    def test_new_mining_is_deterministic_and_has_forward_nondecreasing_bodies(self):
        facts = [(0, 1, 1, 1), (1, 1, 2, 1), (0, 0, 2, 2)]
        args = facts, {0: "A", 1: "A", 2: "A"}, {0: "r", 1: "s"}, {"s": {"A->A": 1}}
        first = RuleMiner(*args).discover(walks=30)
        second = RuleMiner(*args).discover(walks=30)
        self.assertEqual(first, second)
        self.assertIn([1, 1], [r["body_rels"] for r in first["0"]])

    def test_bounded_semantic_score_and_zero_components(self):
        scores = semantic_scores({1: 0.6, 2: 1, 3: 0}, {1: 0.5, 2: 1, 3: 0.5}, 1)
        self.assertAlmostEqual(scores[1], 0.4)
        self.assertAlmostEqual(scores[3], 0.25)
        self.assertEqual(semantic_scores({1: 1}, {1: 0}, 1), {1: 0})
        self.assertEqual(semantic_scores({1: 0}, {1: 2}, 1), {1: 1})

    def test_complete_chain_temporal_direction_and_unlinked_endpoint(self):
        facts = [(0, 1, 1, 1), (1, 2, 2, 1), (1, 2, 3, 4), (0, 1, 4, 4)]
        retriever = Retriever(facts, {"0": [rule([1, 2])]}, {}, {0: "r", 1: "u", 2: "v"}, {})
        chains = retriever.retrieve(0, 0, 0, 4, 1, 1, 1)
        chain = next(c for c in chains if c.endpoint == 2)
        self.assertEqual(chain.channels[1:], (0, 1))
        self.assertEqual(len(chain.facts), 2)
        self.assertNotIn(3, [c.endpoint for c in chains])
        self.assertNotIn(4, [c.endpoint for c in chains])
        self.assertEqual(retriever.retrieve(0, 0, 0, 1, 1, 1, 1), [])
        backwards = Retriever(facts, {"3": [rule([5, 4], head=3)]}, {}, {0: "r", 1: "u", 2: "v"}, {})
        inverse = next(c for c in backwards.retrieve(2, 0, 1, 4, 1, 1, 1) if c.endpoint == 0)
        self.assertEqual(inverse.facts, ((2, 5, 1, 1), (1, 4, 0, 1)))

    def test_table4_semantic_definition_controls_use_the_same_pool(self):
        frequencies, counts = {1: 1, 2: 0.5, 3: 0}, {1: 0, 2: 2, 3: 1}
        self.assertEqual(semantic_scores(frequencies, counts, 1, "type_frequency"), {1: 1, 2: .5, 3: 0})
        self.assertEqual(semantic_scores(frequencies, counts, 1, "recurrence"), {1: 0, 2: 1, 3: .5})
        self.assertEqual(semantic_scores(frequencies, counts, 1, "product"), {1: 0, 2: 1, 3: 0})
        bounded = semantic_scores(frequencies, counts, 1)
        self.assertEqual(bounded[1], 0)
        self.assertAlmostEqual(bounded[3], 1 / 3)
        with self.assertRaises(ValueError):
            semantic_scores(frequencies, counts, 1, "unsupported")

    def test_broken_connectivity_order_and_repeated_variable_bindings(self):
        facts = [(0, 1, 1, 2), (1, 2, 2, 1), (3, 2, 2, 3)]
        rels = {0: "r", 1: "u", 2: "v"}
        r = Retriever(facts, {"0": [rule([1, 2])]}, {}, rels, {})
        self.assertNotIn(2, [c.endpoint for c in r.retrieve(0, 0, 0, 4, 1, 1, 1)])
        r = Retriever([(0, 1, 1, 1), (1, 2, 2, 2)], {"0": [rule([1, 2], groups=[[0, 2]])]}, {}, rels, {})
        self.assertNotIn(2, [c.endpoint for c in r.retrieve(0, 0, 0, 4, 1, 1, 1)])

    def test_logical_maximum_and_same_chain_fusion(self):
        facts = [(0, 0, 2, 9), (0, 1, 1, 1), (1, 2, 2, 2)]
        rules = {"0": [rule([1, 2], 0.4), rule([1, 2], 0.8)]}
        r = Retriever(facts, rules, {}, {0: "r", 1: "u", 2: "v"}, {})
        chain = next(c for c in r.retrieve(0, 0, 0, 10, 1, 1, 1) if c.endpoint == 2)
        self.assertEqual(len(chain.facts), 2)
        self.assertLess(chain.channels[0], 1)
        self.assertEqual(chain.channels[2], 1)
        self.assertLess(chain.score, 3)  # Independent channel maxima would invent score 3.

    def test_recurrence_counts_observations_across_relations_not_rule_multiplicity(self):
        facts = [(0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 2, 2), (0, 1, 2, 5)]
        r = Retriever(facts, {}, {}, {0: "r", 1: "s"}, {})
        pool, counts = r.python_pool(0, 0, 0, 4)
        self.assertEqual(dict(counts), {1: 2, 2: 1})
        self.assertEqual(len(pool), 2)

    def test_deterministic_endpoint_and_witness_ties(self):
        facts = [(0, 1, 2, 2), (0, 1, 1, 2), (0, 0, 1, 2)]
        r = Retriever(facts, {}, {}, {0: "r", 1: "s"}, {})
        chains = r.retrieve(0, 0, 0, 3, 1, 1, 1)
        self.assertEqual([c.endpoint for c in chains], [1, 2])
        self.assertEqual(chains[0].facts, ((0, 0, 1, 2),))
        self.assertEqual(chains, Retriever(list(reversed(facts)), {}, {}, {0: "r", 1: "s"}, {}).retrieve(0, 0, 0, 3, 1, 1, 1))

    def test_shared_facts_membership_and_whole_chain_packing(self):
        a, b, c = (0, 0, 1, 1), (1, 0, 2, 2), (1, 0, 3, 2)
        chains = [Chain(2, (a, b), (1, 0, 1), 2), Chain(3, (a, c), (1, 0, 1), 1)]
        entities = {i: f"entity{i}" for i in range(4)}
        text = serialize(0, 0, 0, 3, chains, entities, {0: "r"}, {})
        self.assertIn("2:[0,1]", text)
        self.assertIn("3:[0,2]", text)
        self.assertEqual(text.count("1:[0,0,1]"), 1)
        one = serialize(0, 0, 0, 3, chains[:1], entities, {0: "r"}, {})
        prompt, info = pack(0, 0, 0, 3, chains, entities, {0: "r"}, {}, Tokenizer(), len(one) + 2)
        self.assertEqual(info["serialized_endpoints"], [2])
        self.assertNotIn("3:entity3", prompt)
        self.assertIn("2:[0,1]", prompt)
        self.assertEqual(prompt.count("### "), 7)
        inverse = serialize(2, 0, 1, 3, [], entities, {0: "r"}, {})
        self.assertIn("3:[2,1,?]", inverse)
        with self.assertRaises(ValueError):
            pack(0, 0, 0, 3, [], entities, {0: "r"}, {}, Tokenizer(), 1)

    def test_full_gid_numeric_suffix_and_termination(self):
        t = Tokenizer()
        vocab = {"1000": "a", "100": "b", "123": "c"}
        self.assertEqual(decode_sequence(t, 10, 5, vocab), "1000")
        self.assertEqual(decode_sequence(t, 10, 0, vocab), "100")
        self.assertEqual(decode_sequence(t, 10, 4, vocab), "100")
        self.assertIsNone(decode_sequence(t, 4, 5, vocab))
        self.assertIsNone(decode_sequence(t, 10, 5, {"100": "b"}))

    def test_joint_probability_includes_terminator_and_duplicate_paths_use_maximum(self):
        best, vocabulary = {}, {"123": "x", "1": "y"}
        observe_pairs(best, Tokenizer(), vocabulary, 1, math.log(0.2), [(2, math.log(0.9)), (0, math.log(0.1))])
        observe_pairs(best, Tokenizer(), vocabulary, 3, math.log(0.8), [(0, math.log(0.1))])
        self.assertAlmostEqual(math.exp(best["123"][0]), 0.18)
        self.assertEqual(best["123"][1], (1, 2))
        self.assertAlmostEqual(math.exp(best["1"][0]), 0.02)

    def test_filter_before_top10_gold_retention_duplicate_gid_and_token_pair_ties(self):
        vocabulary = {str(i): str(i) for i in range(12)}
        traces = [{"gid": str(i), "token_ids": [i, 0], "log_joint_probability": -i} for i in reversed(range(12))]
        traces += [{"gid": "10", "token_ids": [10, 1], "log_joint_probability": -11}]
        best = consolidate(traces, vocabulary)
        result = rank_record("q", [r["gid"] for r in best], vocabulary, "10", ["0", "10"], best)
        self.assertIsNone(result["raw_rank"])
        self.assertEqual(result["filtered_rank"], 10)
        self.assertEqual(next(r for r in best if r["gid"] == "10")["token_ids"], [10, 0])
        tie = consolidate([{"gid": "1", "token_ids": [2, 1], "log_joint_probability": -1},
                           {"gid": "1", "token_ids": [1, 2], "log_joint_probability": -1}], vocabulary)
        self.assertEqual(tie[0]["token_ids"], [1, 2])

    def test_missing_predictions_stay_in_denominator_and_bad_query_ids_fail(self):
        records = [{"query_id": "hit", "output": "1", "filters": []}, {"query_id": "miss", "output": "1", "filters": []}]
        meta = {"prepared_sha256": "hash", "entity_vocabulary": {"1": "1"}, "scope": "complete", "full_queries": 2}
        prediction = {"query_id": "hit", "candidate_sequences": [{"gid": "1", "token_ids": [1, 0], "log_joint_probability": -1}],
                      "prepared_sha256": "hash", "scoring_protocol": SCORING_PROTOCOL}
        report = replay(records, meta, [prediction])
        self.assertEqual(report["metrics"]["filter hit@1"], 0.5)
        self.assertEqual(report["missing_predictions"], 1)
        self.assertEqual(replay(records, meta, [])["metrics"]["filter hit@10"], 0)
        with self.assertRaises(ValueError):
            replay(records, meta, [prediction, prediction])
        with self.assertRaises(ValueError):
            replay(records, meta, [dict(prediction, scoring_protocol="old")])

    def test_standalone_preparation_cache_fingerprints_and_full_query_count(self):
        scratch = ROOT / "outputs"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            asset = root / "assets/ICEWS14"
            asset.mkdir(parents=True)
            (asset / "entity2id.txt").write_text("A\t0\nB\t2\n", encoding="utf-8")  # sparse entity IDs are preserved
            (asset / "relation2id.txt").write_text("r\t0\n", encoding="utf-8")
            write_json(asset / "entity2label_clustered_final.json", {"A": "Person", "B": "Person"})
            write_json(asset / "relation_type_percentages.json", {"r": {"Person->Person": 100}})
            write_json(asset / "rules/joint.json", {})
            write_json(root / "configs/protocol.json", {"sampling_seed": 42, "endpoint_budget": 50, "cutoff_len": 1536,
                        "datasets": {"ICEWS14": {"alpha": 1, "lambda": 1, "beta": 1}}})
            checks = {}
            for split, facts in {"train": [(0, 0, 2, 1)], "valid": [(0, 0, 2, 2)], "test": [(0, 0, 2, 3)]}.items():
                folder = root / "data/ICEWS14"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / (split + ".txt")).write_text("".join("\t".join(map(str, f)) + "\n" for f in facts), encoding="utf-8")
                checks[split] = {"rows": 1, "canonical_sha256": canonical_hash(facts)}
            write_json(root / "configs/benchmark_checksums.json", {"ICEWS14": checks})
            output = root / "processed/ICEWS14-test.json"
            meta = prepare(root, "ICEWS14", "test", Tokenizer(), output, root / "data")
            records, loaded = load_prepared(root, output, Tokenizer())
            self.assertEqual((meta["queries"], meta["full_queries"]), (2, 2))
            self.assertEqual({r["output"] for r in records}, {"0", "2"})
            (asset / "relation2id.txt").write_text("changed\t0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prepared(root, output)

    @unittest.skipUnless(importlib.util.find_spec("grounded_chain_native"), "optional native extension not built")
    def test_native_matches_exhaustive_python_on_small_random_graphs(self):
        rng = random.Random(9)
        rels, types = {0: "r", 1: "s"}, {i: str(i % 2) for i in range(4)}
        rules = {"0": [rule([1, 1]), rule([1, 3, 0], 0.4), rule([0, 0], 0.3, groups=[[0, 2]])]}
        for _ in range(12):
            facts = [(rng.randrange(4), rng.randrange(2), rng.randrange(4), rng.randrange(1, 5)) for _ in range(20)]
            py = Retriever(facts, rules, types, rels, {})
            native = Retriever(facts, rules, types, rels, {}, "native")
            for known in range(4):
                for direction in (0, 1):
                    a = py.retrieve(known, 0, direction, 5, 1, 1, 1)
                    b = native.retrieve(known, 0, direction, 5, 1, 1, 1)
                    self.assertEqual([c.endpoint for c in a], [c.endpoint for c in b])
                    for x, y in zip(a, b):
                        self.assertEqual(x.facts, y.facts)
                        self.assertAlmostEqual(x.score, y.score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
