"""Resolution choices belong to rollout batches, never to replay updates."""
import math
import random


def balanced_dance_picks(seed, step, replica, replicas, batch, states, steps, beta, mb, parts):
    """Redistribute the original global draw before pairing times with local prompts."""
    import numpy as np

    draws = [np.random.default_rng([seed, r, step, 52]).beta(*beta, size=(batch, states))
             for r in range(replicas)]
    picks = np.minimum((np.concatenate(draws) * steps).astype(np.int64), steps - 1)
    depth = picks.max(axis=1)
    stripes = np.argsort(depth, kind="stable").reshape(batch, replicas)
    per_part = math.ceil(batch / parts)
    cost = np.zeros(replicas, dtype=np.int64)
    local = np.empty((batch, states), dtype=np.int64)
    rng = np.random.default_rng([seed, step, 53])
    tie_order = rng.permutation(replicas)
    for i, stripe in enumerate(stripes):
        # Local sorting makes the final row of each rollout microbatch its forward count.
        offset = i % per_part + 1
        if offset % mb == 0 or offset == per_part or i + 1 == batch:
            ranks = tie_order[np.argsort(cost[tie_order], kind="stable")]
            stripe = stripe[::-1]
            cost[ranks] += depth[stripe]
        else:
            ranks = tie_order
        local[i] = picks[stripe[np.argmax(ranks == replica)]]
    # Randomize time-to-prompt assignment before the usual joint local sort.
    return local[np.random.default_rng([seed, replica, step, 54]).permutation(batch)]


def validate(args):
    resolutions = args.opd_resolutions
    microbatches = getattr(args, "opd_res_mb", ())
    if microbatches and (not resolutions or len(microbatches) != len(resolutions) or
                         any(v <= 0 for v in microbatches)):
        raise ValueError("--opd-res-mb needs one positive image microbatch per OPD resolution")
    if not resolutions:
        if getattr(args, "opd_token_batch", False):
            raise ValueError("--opd-token-batch needs --opd-resolutions")
        if args.opd_res_weights or args.opd_res_shifts:
            raise ValueError("--opd-res-weights / --opd-res-shifts need --opd-resolutions")
        return
    if not args.opd or args.pp != 1 or args.multi_res or args.crop_res or args.repa_weight:
        raise ValueError("--opd-resolutions needs OPD, --pp 1, no --multi-res / --crop-res / REPA")
    if len(set(resolutions)) != len(resolutions) or any(r <= 0 or r % 16 for r in resolutions):
        raise ValueError("OPD resolutions must be distinct positive multiples of 16")
    if max(resolutions) != args.res:
        raise ValueError("--res must equal the largest --opd-resolutions value")
    for name in ("opd_res_weights", "opd_res_shifts"):
        values = getattr(args, name)
        if values and (len(values) != len(resolutions) or
                       any(not math.isfinite(v) or v <= 0 for v in values)):
            raise ValueError(f"--{name.replace('_', '-')} needs one positive finite value per resolution")
    if not args.opd_res_shifts and (not math.isfinite(args.t_shift) or args.t_shift <= 0 or
                                   not math.isfinite(args.t_shift_ref) or args.t_shift_ref < 0):
        raise ValueError("Invalid OPD time shift")


def batch_resolution(args, step):
    """All learners select the same size; no global RNG consumption or prefetch side effects."""
    resolutions = getattr(args, "opd_resolutions", ())
    if not resolutions:
        return args.res
    rng = random.Random(args.seed * 1_000_003 + step * 1009 + 739)
    return rng.choices(resolutions, weights=args.opd_res_weights or None, k=1)[0]


def time_shift(args, resolution):
    resolutions = getattr(args, "opd_resolutions", ())
    if resolutions:
        shifts = args.opd_res_shifts
        if shifts:
            return shifts[resolutions.index(resolution)]
        if args.t_shift_ref:
            return resolution / args.t_shift_ref
        return args.t_shift * resolution / args.res
    # Preserve historical single-resolution OPD behaviour.
    return args.t_shift


def batch_multiplier(args, resolution):
    """Whole base batches keep replica / microbatch / OPD-part divisibility intact."""
    return max(1, round((args.res / resolution) ** 2)) if getattr(args, "opd_token_batch", False) else 1


def microbatch_size(args, resolution, default):
    values = getattr(args, "opd_res_mb", ())
    return values[args.opd_resolutions.index(resolution)] if values else default


def validate_microbatches(args, per_pipe, default):
    """Both rollout trajectories and training states must fill complete microbatches."""
    for resolution in args.opd_resolutions:
        n = per_pipe * batch_multiplier(args, resolution)
        mb = microbatch_size(args, resolution, default)
        part = (n + args.opd_parts - 1) // args.opd_parts
        if any(min(part, n - start) % mb for start in range(0, n, part)):
            raise ValueError(f"OPD {resolution}px: {n} images per learner split into "
                             f"{args.opd_parts} parts must fill microbatch {mb}")


def gather_caption_batches(loader, count, has_cond=False):
    """Discard unused images immediately; combine only captions and their source rows."""
    import torch
    ids, rows = [], []
    for _ in range(count):
        got = next(loader)
        ids.append(got[1])
        rows.append(got[2])
        del got
    ids, rows = torch.cat(ids), torch.cat(rows)
    result = (torch.empty(len(ids), 0), ids, rows)
    # next_batch removes loader conditions; OPD uses whole-image conditioning.
    return result + (torch.empty(len(ids), 8),) if has_cond else result
