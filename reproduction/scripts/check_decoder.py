"""Optional CPU-logit integration checks; requires the PyTorch environment."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import torch
from myst_fusion.decoding import (predict_candidates, decode_sequence, rank_record,
                             merge_query_records, SCORING_PROTOCOL)


class DigitTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [138, 139] if text == 'CTX\n' else []

    def decode(self, ids, skip_special_tokens=False):
        pieces = {0: '<eos>', 1: '1', 2: '23', 3: '123', 4: 'x', 5: '0',
                  6: '12', 7: 'junk', 8: '999', 9: '2', 10: '100'}
        return ''.join(pieces.get(i, f'z{i}') for i in ids)


class ToyModel:
    def __init__(self, first=None, second=None):
        self.calls = []
        self.first = torch.full((140,), -20.) if first is None else first
        if first is None:
            self.first[1], self.first[6], self.first[3] = 5., 4., 3.
        self.second = second or {}

    def eval(self): pass

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(1))

    def __call__(self, input_ids, attention_mask, use_cache):
        self.calls.append(input_ids.tolist())
        if input_ids.shape[1] == 2:
            scores = self.first[None]
        else:
            default = torch.full((140,), -20.)
            default[0] = 6.
            scores = torch.stack([self.second.get(int(row[-1]), default) for row in input_ids])
        return SimpleNamespace(logits=scores[:, None, :].expand(-1, input_ids.shape[1], -1))


class ScoringTests(unittest.TestCase):
    def test_full_two_token_id_and_single_token_termination(self):
        tokenizer = DigitTokenizer()
        vocab = {'1000': 'long', '100': 'short', '123': 'other'}
        self.assertEqual(decode_sequence(tokenizer, 10, 5, vocab), '1000')
        self.assertEqual(decode_sequence(tokenizer, 10, 0, vocab), '100')
        self.assertEqual(decode_sequence(tokenizer, 10, 4, vocab), '100')
        self.assertIsNone(decode_sequence(tokenizer, 10, 5, {'100': 'short'}))
        self.assertIsNone(decode_sequence(tokenizer, 4, 5, vocab))

    def test_joint_probability_and_maximum_per_duplicate_gid(self):
        second = torch.full((140,), -20.)
        second[2] = 8.
        model = ToyModel(second={1: second})
        candidates, traces = predict_candidates(model, DigitTokenizer(), 'CTX\n', {'123': 'a', '12': 'b'})
        self.assertEqual(candidates, ['123', '12'])
        self.assertEqual(traces[0]['token_ids'], [1, 2])
        expected = torch.log_softmax(model.first, 0)[1] + torch.log_softmax(second, 0)[2]
        self.assertAlmostEqual(traces[0]['log_joint_probability'], float(expected), places=6)
        self.assertEqual(len({r['gid'] for r in traces}), len(traces))

    def test_both_steps_prune_at_100_before_valid_gid_selection(self):
        class NumericTokenizer(DigitTokenizer):
            def decode(self, ids, skip_special_tokens=False):
                return ''.join('<eos>' if i == 0 else str(i) for i in ids)
        first = torch.arange(140., 0., -1)
        model = ToyModel(first=first, second={i: first for i in range(100)})
        candidates, _ = predict_candidates(model, NumericTokenizer(), 'CTX\n',
                                            {'99': 'valid', '100': 'pruned', '99100': 'pruned_suffix'})
        self.assertEqual(candidates, ['99'])

    def test_second_step_probability_alone_does_not_determine_order(self):
        first = torch.full((140,), -20.)
        first[1], first[6] = 6., 1.
        diffuse = torch.zeros(140)
        sharp = torch.full((140,), -20.); sharp[0] = 10.
        candidates, _ = predict_candidates(ToyModel(first, {1: diffuse, 6: sharp}),
                                            DigitTokenizer(), 'CTX\n', {'1': 'a', '12': 'b'})
        self.assertEqual(candidates, ['1', '12'])

    def test_batch_size_does_not_change_results_and_ties_are_token_sequences(self):
        first = torch.zeros(140)
        args = (DigitTokenizer(), 'CTX\n', {'1': 'a', '12': 'b', '123': 'c'})
        one = predict_candidates(ToyModel(first), *args, second_step_batch_size=1)
        eight = predict_candidates(ToyModel(first), *args, second_step_batch_size=8)
        self.assertEqual(one, eight)
        self.assertEqual([t['token_ids'][0] for t in one[1]], [1, 3, 6])

    def test_filtered_order_full_query_denominator_and_protocol_guard(self):
        vocab = {'0': 'a', '1': 'b', '2': 'c'}
        hit = rank_record('hit', ['2', '1', '0'], vocab, '0', ['a', 'b', 'c'])
        miss = rank_record('miss', [], vocab, '0', [])
        self.assertEqual((hit['raw_rank'], hit['filtered_rank']), (3, 1))
        metrics, _ = merge_query_records([miss, hit, hit], ['hit', 'miss'])
        self.assertEqual(metrics['filter hit@1'], .5)
        self.assertEqual(metrics['raw hit@10'], .5)
        with self.assertRaises(ValueError): merge_query_records([hit], ['hit', 'miss'])
        with self.assertRaises(ValueError):
            merge_query_records([dict(hit, scoring_protocol='next-token-top10-global-gid-v1')], ['hit'])

    def test_time_filter_precedes_ranking_and_top10(self):
        vocab = {str(i): i for i in range(12)}
        # Deliberately supply an unordered pool; answer 10 is raw rank 11.
        candidates = [str(i) for i in reversed(range(12))]
        traces = [{'gid': gid, 'log_joint_probability': -int(gid),
                   'token_ids': [int(gid), 0]} for gid in candidates]
        record = rank_record('q', candidates, vocab, '10', {0, 10}, traces)
        self.assertIsNone(record['raw_rank'])
        self.assertEqual(record['filtered_rank'], 10)
        self.assertEqual(record['filtered_predictions'], list(range(1, 11)))

    def test_decoder_keeps_candidates_beyond_ten_for_metric_filtering(self):
        class NumericTokenizer(DigitTokenizer):
            def decode(self, ids, skip_special_tokens=False):
                return ''.join('<eos>' if i == 0 else str(i) for i in ids)
        vocab = {str(i): i for i in range(1, 13)}
        first = torch.arange(140., 0., -1)
        candidates, traces = predict_candidates(ToyModel(first), NumericTokenizer(), 'CTX\n', vocab)
        self.assertEqual(len(candidates), 12)
        # The searched pool is deliberately unranked. Pick raw rank 11 using
        # joint scores, rather than assuming dictionary insertion order is rank.
        ranked = sorted(traces, key=lambda t: (-t['log_joint_probability'], tuple(t['token_ids'])))
        answer = ranked[10]['gid']
        record = rank_record('q', candidates, vocab, answer, {vocab[ranked[0]['gid']]}, traces)
        self.assertIsNone(record['raw_rank'])
        self.assertEqual(record['filtered_rank'], 10)

    def test_empty_overbudget_nonfinite_and_bad_batch_inputs(self):
        for prompt, cutoff in [('', 10), ('CTX\n', 3)]:
            with self.assertRaises(ValueError):
                predict_candidates(ToyModel(), DigitTokenizer(), prompt, {'1': 'a'}, cutoff)
        with self.assertRaises(ValueError):
            predict_candidates(ToyModel(torch.full((140,), float('nan'))), DigitTokenizer(), 'CTX\n', {'1': 'a'})
        with self.assertRaises(ValueError):
            predict_candidates(ToyModel(second={1: torch.full((140,), float('nan'))}), DigitTokenizer(), 'CTX\n', {'1': 'a'})
        with self.assertRaises(ValueError):
            predict_candidates(ToyModel(), DigitTokenizer(), 'CTX\n', {'1': 'a'}, second_step_batch_size=0)


if __name__ == '__main__': unittest.main(verbosity=2)
