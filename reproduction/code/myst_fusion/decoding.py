"""Two-step 100-by-100 search, joint-probability ranking and fixed GIDs."""
from .prompt import encode


SCORING_PROTOCOL = 'two-step-top100x100-joint-gid-filter-before-rank-v2'
TOP_TOKENS = 10
FIRST_STEP_TOP_K = 100
SECOND_STEP_TOP_K = 100


def decode_candidates(tokenizer, token_ids, vocabulary):
    """Map only complete decoded GIDs; preserve order without replenishment."""
    candidates, seen = [], set()
    for token_id in token_ids:
        gid = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if gid in vocabulary and vocabulary[gid] is not None and gid not in seen:
            candidates.append(gid)
            seen.add(gid)
    return candidates


def decode_sequence(tokenizer, first, second, vocabulary):
    """A nonnumeric second token terminates the first token's complete ID."""
    suffix = tokenizer.decode([second], skip_special_tokens=False).strip()
    numeric_suffix = second != getattr(tokenizer, 'eos_token_id', None) and bool(suffix) and suffix.isascii() and suffix.isdigit()
    ids = [first, second] if numeric_suffix else [first]
    gid = tokenizer.decode(ids, skip_special_tokens=False).strip()
    if gid in vocabulary and vocabulary[gid] is not None:
        return gid
    return None


def observe_pairs(best, tokenizer, vocabulary, first, first_log_probability, second_choices):
    """Retain the maximum actual joint score for each complete mapped GID."""
    for second, second_log_probability in second_choices:
        gid = decode_sequence(tokenizer, first, second, vocabulary)
        if gid is None:
            continue
        score = float(first_log_probability + second_log_probability)
        token_ids = (first, second)
        previous = best.get(gid)
        if previous is None or (-score, token_ids) < (-previous[0], previous[1]):
            best[gid] = (score, token_ids)


def predict_candidates(model, tokenizer, prompt, vocabulary, cutoff_len=1536,
                       second_step_batch_size=8):
    """Score the complete searched pool without final entity ranking or truncation."""
    import torch
    prefix = encode(tokenizer, prompt)
    if not prefix:
        raise ValueError('Empty forecasting context')
    if len(prefix) + 2 > cutoff_len:
        raise ValueError('Prompt exceeds cutoff; implicit truncation is forbidden')
    if second_step_batch_size < 1:
        raise ValueError('Second-step batch size must be positive')
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prefix], device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite next-token logits')
        # Normalize over the full vocabulary, before pruning either step.
        first_log_probs = torch.log_softmax(logits, dim=-1)
        first_ids = torch.argsort(logits, descending=True, stable=True)[:FIRST_STEP_TOP_K].tolist()
        best = {}
        for start in range(0, len(first_ids), second_step_batch_size):
            batch = first_ids[start:start + second_step_batch_size]
            inputs = torch.tensor([prefix + [token] for token in batch], device=device)
            second_logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  use_cache=False).logits[:, -1].float()
            if not torch.isfinite(second_logits).all():
                raise ValueError('Nonfinite second-step logits')
            second_log_probs = torch.log_softmax(second_logits, dim=-1)
            for row, first in enumerate(batch):
                second_ids = torch.argsort(second_logits[row], descending=True,
                                           stable=True)[:SECOND_STEP_TOP_K].tolist()
                observe_pairs(best, tokenizer, vocabulary, first, float(first_log_probs[first]),
                              [(second, float(second_log_probs[row, second])) for second in second_ids])
        candidates = list(best)
        traces = [{'gid': gid, 'token_ids': list(best[gid][1]),
                   'log_joint_probability': best[gid][0]} for gid in candidates]
        # Keep the complete searched entity pool. Metric filtering must happen
        # before the final entity ranking and Top-10 selection.
    return candidates, traces


def rank_record(query_id, candidates, vocabulary, answer_gid, filters, traces=None):
    if answer_gid not in vocabulary:
        raise ValueError('Label must be a valid global entity GID')
    if len(candidates) != len(set(candidates)) or any(gid not in vocabulary for gid in candidates):
        raise ValueError('Candidates must be distinct valid GIDs')
    if traces is not None:
        if [row['gid'] for row in traces] != candidates:
            raise ValueError('Candidate scores must match the complete searched pool')
        scores = {row['gid']: row for row in traces}
        order = lambda gid: (-scores[gid]['log_joint_probability'], tuple(scores[gid]['token_ids']))
    else:
        positions = {gid: i for i, gid in enumerate(candidates)}
        order = positions.__getitem__
    answer = vocabulary[answer_gid]
    excluded = set(filters) - {answer}
    filtered = [gid for gid in candidates if vocabulary[gid] not in excluded]
    filtered = sorted(filtered, key=order)[:TOP_TOKENS]
    rank = filtered.index(answer_gid) + 1 if answer_gid in filtered else None
    # Raw ranking is a separate diagnostic, never the input to filtered Hits.
    raw = sorted(candidates, key=order)[:TOP_TOKENS]
    raw_rank = raw.index(answer_gid) + 1 if answer_gid in raw else None
    return {'query_id': query_id, 'raw_rank': raw_rank, 'filtered_rank': rank,
            'predictions': [vocabulary[gid] for gid in raw],
            'filtered_predictions': [vocabulary[gid] for gid in filtered],
            'scoring_protocol': SCORING_PROTOCOL}


def merge_query_records(records, expected_ids):
    unique = {}
    for record in records:
        if record.get('scoring_protocol') != SCORING_PROTOCOL:
            raise ValueError('Incompatible scoring protocol in query records')
        key = record['query_id']
        if key in unique and unique[key] != record:
            raise ValueError('Conflicting distributed records for the same query')
        for field in ('raw_rank', 'filtered_rank'):
            rank = record[field]
            if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= TOP_TOKENS):
                raise ValueError('Invalid candidate rank')
        unique[key] = record
    if set(unique) != set(expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Missing, extra or duplicate query IDs in evaluation')
    ordered = [unique[q] for q in expected_ids]
    if not ordered:
        raise ValueError('Cannot evaluate an empty query set')
    metrics = {}
    for label, field in [('raw', 'raw_rank'), ('filter', 'filtered_rank')]:
        for k in (1, 3, 10):
            metrics[f'{label} hit@{k}'] = sum(r[field] is not None and r[field] <= k for r in ordered) / len(ordered)
    return metrics, ordered
