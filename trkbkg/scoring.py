"""Shared model-scoring helpers (used by training AND evaluation).

Since the 2026-07-15 reorganization the training LOOPS live inside the
3xx scripts themselves, readable top to bottom; what stays here is only
what several scripts genuinely share: scoring a whole fold in batches,
for per-hit models and for event-context models (which must be scored
event-batched so pooling only ever sees hits of one event, exactly as
in deployment).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import load_fold, load_meta
from .evaluate import rejection_at_efficiency, roc_auc
from .models import build_model


def event_ranges(evt: np.ndarray):
    """(starts, lengths) of each event's contiguous hit block.

    Requires hits grouped by event (true for our datasets: file order is
    event order and the split preserves it) — verified by the caller.
    """
    change = np.flatnonzero(np.diff(evt.astype(np.int64))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(evt)]])
    return starts, ends - starts


def score_fold(model, X, sys_idx, evt=None, hit_batch: int = 1_000_000,
               events_per_batch: int = 128) -> np.ndarray:
    """Scores for a whole fold. X/sys_idx may live on CPU or GPU; slices are
    moved to the model's device per batch (cheap: ~25 MB per 1M hits).

    Event-context models are scored event-batched so that pooling only ever
    sees hits of one event — identical to how they run in deployment.
    """
    model.eval()
    mdev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        if hasattr(model, "forward_batched"):
            starts, lengths = event_ranges(evt)
            dev = X.device
            for i in range(0, len(starts), events_per_batch):
                s, l = starts[i:i + events_per_batch], lengths[i:i + events_per_batch]
                idx = torch.cat([torch.arange(a, a + n, device=dev)
                                 for a, n in zip(s, l)])
                seg = torch.repeat_interleave(
                    torch.arange(len(s), device=mdev),
                    torch.as_tensor(l, device=mdev))
                logits = model.forward_batched(X[idx].to(mdev),
                                               sys_idx[idx].to(mdev),
                                               seg, len(s))
                out.append(torch.sigmoid(logits).cpu().numpy())
        else:
            for i in range(0, len(X), hit_batch):
                out.append(torch.sigmoid(
                    model(X[i:i + hit_batch].to(mdev),
                          sys_idx[i:i + hit_batch].to(mdev))).cpu().numpy())
    return np.concatenate(out)



# backwards-compat alias used by earlier scripts
def score_in_batches(model, X, sys_idx, batch: int = 1_000_000):
    return score_fold(model, X, sys_idx, hit_batch=batch)
