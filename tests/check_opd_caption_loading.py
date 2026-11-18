"""Caption-only loading preserves sampling; preparation overlaps snapshot I/O."""
import tempfile
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import torch

from aurora.data import ShardStore, TrainStream
from aurora.opd_actor import RolloutClient


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    np.save(root / 's.off.npy', np.arange(0, 65, 2))
    tokens = np.arange(64 * 4).reshape(64, 4)
    np.save(root / 's.tok.npy', tokens)
    np.save(root / 's.tok_pl.npy', tokens + 1000)
    np.save(root / 's.lat.npy', np.zeros((32, 2, 2, 2), dtype=np.float16))
    def store(images=False):
        return ShardStore(tmp, ['s'], null_row=np.zeros(4), alt_suffix='pl',
                          alt_prob=.5, img_res=1024 if images else 0)
    reference = iter(TrainStream(store(), 4, 1, 2, 87462, rows=True, sort_pad=0))
    # No image files exist: any image mmap/decode would fail this test.
    captions = store(images=True)
    fast = iter(TrainStream(captions, 4, 1, 2, 87462, rows=True,
                            sort_pad=0, captions_only=True))
    for _ in range(12):
        old, new = next(reference), next(fast)
        assert new[0].shape == (4, 0)
        assert torch.equal(old[1], new[1]) and torch.equal(old[2], new[2])
    assert captions._lat is None
    resumed = iter(TrainStream(store(images=True), 4, 1, 2, 87462, start_micro=12,
                               rows=True, sort_pad=0, captions_only=True))
    assert torch.equal(next(fast)[1], next(resumed)[1])
print('PASS: caption IDs, PL/EN choice, ordering and resume preserved; no image files opened')

client = RolloutClient(1, None)
io_started, release_io, prep_started, release_prep = [Event() for _ in range(4)]
def snapshot():
    io_started.set()
    assert release_io.wait(10)
def prepare():
    prep_started.set()
    assert release_prep.wait(10)
    return {'actor_step': 8}
try:
    client.snapshot_pending = client.pool.submit(snapshot)
    assert io_started.wait(10)
    with patch.object(client, '_request', side_effect=lambda payload: payload):
        client.request_prepared(prepare)
        assert prep_started.wait(10)  # preparation starts despite blocked snapshot I/O
        assert not client.pending.done()  # caller remains free to train
        try:
            client.request_prepared(prepare)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unbounded caption queue')
        release_io.set()
        release_prep.set()
        assert client.collect() == {'actor_step': 8}
        def fail():
            raise ValueError('caption read failed')
        client.request_prepared(fail)
        try:
            client.collect()
        except ValueError as error:
            assert str(error) == 'caption read failed'
        else:
            raise AssertionError('Worker exception hidden')
finally:
    release_io.set()
    release_prep.set()
    client.pool.shutdown(wait=True)
    client.prepare_pool.shutdown(wait=True)
print('PASS: bounded asynchronous preparation overlaps snapshot transfer and propagates errors')
