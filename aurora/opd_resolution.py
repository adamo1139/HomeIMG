"""Resolution choices belong to rollout batches, never to replay updates."""
import math
import random


def validate(args):
    resolutions = args.opd_resolutions
    if not resolutions:
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
