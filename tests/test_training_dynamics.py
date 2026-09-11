"""CPU-only tests: legacy LR, controlled tail experiment and pooled diagnostics."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_dynamics import cosine_learning_rate, dynamics_row


class TrainingDynamicsTests(unittest.TestCase):
    def test_mid_ramp_preserves_early_rounds_and_floor(self):
        for r in range(1, 61):
            self.assertEqual(cosine_learning_rate(r, .1, 300, mid_factor=.5),
                             cosine_learning_rate(r, .1, 300))
        self.assertAlmostEqual(cosine_learning_rate(75, .1, 300, mid_factor=.5),
                               .75 * cosine_learning_rate(75, .1, 300))
        self.assertAlmostEqual(cosine_learning_rate(90, .1, 300, mid_factor=.5),
                               .5 * cosine_learning_rate(90, .1, 300))
        self.assertEqual(cosine_learning_rate(300, .1, 300, mid_factor=.5), .0001)

    def test_invalid_mid_ramp_rejected(self):
        for start, end, factor in ((90, 60, .5), (-1, 60, .5), (60, 90, 0),
                                   (60, 90, float('nan'))):
            with self.assertRaises(ValueError):
                cosine_learning_rate(100, .1, 300, mid_start=start, mid_end=end, mid_factor=factor)

    def test_default_matches_all_legacy_rounds(self):
        for budget in (300, 500):
            for r in range(1, budget + 1):
                old = max(0.1 * 0.5 * (1 + math.cos(math.pi * r / max(1, budget))), 1e-4)
                self.assertEqual(cosine_learning_rate(r, 0.1, budget), old)

    def test_higher_floor_changes_only_tail(self):
        changes = [r for r in range(1, 301) if
                   cosine_learning_rate(r, .1, 300, .001) != cosine_learning_rate(r, .1, 300)]
        self.assertEqual(changes, list(range(281, 301)))
        self.assertEqual(cosine_learning_rate(300, .1, 300, .001), .001)

    def test_no_rebound_after_budget(self):
        for r in (301, 400, 600, 1000):
            self.assertEqual(cosine_learning_rate(r, .1, 300), .0001)

    def test_invalid_lr_rejected(self):
        for initial, rounds, floor in ((.1, 0, .001), (.1, 300, 0), (.1, 300, .2),
                                      (float('nan'), 300, .001), (.1, 300, float('inf'))):
            with self.assertRaises(ValueError):
                cosine_learning_rate(1, initial, rounds, floor)

    def test_pooled_mass_and_aux_loss(self):
        args = SimpleNamespace(lambda_A=1., lambda_B=1., lambda_proto=.1)
        small = dict(cnt_u=100, cnt_a=50, cnt_b=25, cnt_c=25, n_batches=1,
                     lr=.08, gate=0., aux_scale=.2, lambda_A_scale=1., lambda_B_scale=1.,
                     dyn_a_mass=50., dyn_a_wrong=10., dyn_b_mass=4., dyn_b_wrong=2.,
                     L_sup=1., L_A=.5, L_B=.4, L_proto=2., loss=1.62)
        large = dict(small, cnt_u=300, cnt_a=30, cnt_b=120, cnt_c=150, n_batches=3,
                     dyn_a_mass=30., dyn_a_wrong=3., dyn_b_mass=18., dyn_b_wrong=6.)
        row = dynamics_row(70, 1, [small, large], args)
        self.assertEqual(row['a_ratio'], .2)  # Not unweighted mean of .5 and .1.
        self.assertEqual(row['a_effective_mass_per_u'], .2)
        self.assertEqual(row['a_wrong_mass_per_u'], 13 / 400)
        self.assertAlmostEqual(row['b_mean_weight'], 22 / 145)
        self.assertAlmostEqual(row['L_B_effective'], .08)
        self.assertAlmostEqual(row['L_proto_effective'], .04)
        self.assertAlmostEqual(row['loss_reconstructed'], row['loss_observed'])

    def test_aux_zero_has_no_b_or_proto_loss(self):
        args = SimpleNamespace(lambda_A=1., lambda_B=1., lambda_proto=.1)
        row = dynamics_row(1, 1, [dict(cnt_u=100, cnt_a=10, cnt_b=0, cnt_c=90,
            n_batches=1, lr=.1, gate=0., aux_scale=0., lambda_A_scale=1., lambda_B_scale=1.,
            L_sup=1., L_A=.1, L_B=0., L_proto=5., loss=1.1)], args)
        self.assertEqual(row['L_proto_effective'], 0)
        self.assertEqual(row['b_mean_weight'], 0)
        self.assertAlmostEqual(row['loss_reconstructed'], 1.1)


if __name__ == '__main__':
    unittest.main()
