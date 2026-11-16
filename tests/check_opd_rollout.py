"""CPU regression checks for queried states and skipped OPD forwards.
Run: python tests/check_opd_rollout.py
Extract only the rollout function to avoid distributed/CUDA import dependencies.
"""
import ast
from pathlib import Path
import unittest
import torch

source = Path(__file__).resolve().parents[1] / 'train_pp.py'
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'opd_rollout')
namespace = {'torch': torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
rollout = namespace['opd_rollout']

class Stage:
    def __init__(self):
        self.text_cache = None
        self.calls = 0
        self.encodes = 0

    def te(self, ids, lm_states=None):
        self.encodes += 1
        text = ids.float().mean(1)
        if lm_states is not None:
            text = text + lm_states.mean((1, 2))
        return text, None

    def __call__(self, z, t, ids):
        self.calls += 1
        return z.sin() * 0.2 + t[:, None, None, None] + self.text_cache[0][:, None, None, None] * 0.01


def reference(stage, z, ids, lm, mb, ts, picks, widths):
    chunks = []
    for j in range(0, len(z), mb):
        w = widths[j // mb] if widths is not None else ids.shape[1]
        ids_j = ids[j:j+mb, :w]
        cache = stage.te(ids_j, lm_states=None if lm is None else lm[j:j+mb, :w])
        chunks.append((slice(j, j+mb), cache, ids_j))
    out = torch.empty(*picks.shape, *z.shape[1:])
    for k in range(len(ts)-1):
        b, r = (picks == k).nonzero(as_tuple=True)
        out[b, r] = z[b]
        velocities = []
        for sl, cache, ids_j in chunks:
            stage.text_cache = cache
            velocities.append(stage(z[sl], ts[k].expand(len(ids_j)), ids_j).float())
        z = z + (ts[k+1]-ts[k]) * torch.cat(velocities)
    return out


class RolloutTests(unittest.TestCase):
    def test_exact_states_and_call_counts(self):
        torch.manual_seed(41)
        z = torch.randn(5, 2, 3, 3)
        original = z.clone()
        ids = torch.randint(0, 20, (5, 7))
        lm = torch.randn(5, 7, 4)
        u = torch.linspace(1, 0, 17)
        ts = 6*u/(1+5*u)
        for mb in (1, 2, 3):
            widths = [7 - j for j in range((5+mb-1)//mb)]
            for states in (None, lm):
                for picks in (torch.zeros(5, 1, dtype=torch.long),
                              torch.tensor([[0], [1], [7], [15], [3]]),
                              torch.tensor([[0, 15, 0], [1, 4, 2], [7, 7, 2], [15, 3, 0], [2, 1, 3]])):
                    with self.subTest(mb=mb, lm=states is not None, picks=picks.tolist()):
                        stage = Stage()
                        actual = rollout(stage, z, ids, states, mb, ts, picks, widths)
                        expected = reference(Stage(), z, ids, states, mb, ts, picks, widths)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(z, original, rtol=0, atol=0)
                        self.assertIsNone(stage.text_cache)
                        self.assertEqual(stage.calls, sum(int(picks[j:j+mb].max()) for j in range(0, 5, mb)))
                        self.assertEqual(stage.encodes, (5+mb-1)//mb)

    def test_cache_cleared_on_error(self):
        class Failing(Stage):
            def __call__(self, *args):
                raise RuntimeError('test failure')
        stage = Failing()
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            rollout(stage, torch.zeros(1, 1, 1, 1), torch.ones(1, 2), None, 1,
                    torch.linspace(1, 0, 3), torch.tensor([[1]]))
        self.assertIsNone(stage.text_cache)

if __name__ == '__main__':
    unittest.main()
