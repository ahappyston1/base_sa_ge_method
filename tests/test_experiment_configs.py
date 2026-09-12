"""Lightweight checks for scalar YAML experiment isolation and CLI guards."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import args_parser
from training_dynamics import cosine_learning_rate


def config(name):
    # These checked-in recipes contain only scalar mappings, no YAML nesting.
    result = {}
    for line in (ROOT / 'configs' / ('experiment_' + name + '.yaml')).read_text(encoding='utf-8').splitlines():
        text = line.split('#', 1)[0].strip()
        if not text:
            continue
        key, value = text.split(':', 1)
        if key in result:
            raise AssertionError('Duplicate recipe key: ' + key)
        result[key] = value.strip()
    return result


class ExperimentConfigTests(unittest.TestCase):
    def test_followup_recipes_isolate_each_change(self):
        base = config('high_lr')
        for name, changed, mode, lr, rounds in (
            ('trusted_lr015', {'pp_geom_mode'}, 'trusted', '0.15', '300'),
            ('legacy_lr015_500', {'max_rounds'}, 'legacy', '0.15', '500'),
            ('legacy_lr020', {'lr_local_training'}, 'legacy', '0.20', '300'),
        ):
            with self.subTest(recipe=name):
                experiment = config(name)
                self.assertEqual(set(base), set(experiment))
                self.assertEqual({k for k in base if base[k] != experiment[k]}, changed)
                self.assertEqual(experiment['pp_geom_mode'], mode)
                self.assertEqual(experiment['lr_local_training'], lr)
                self.assertEqual(experiment['max_rounds'], rounds)

    def test_long_recipe_stretches_lr_and_phase_budget(self):
        recipe = config('legacy_lr015_500')
        rounds = int(recipe['max_rounds'])
        self.assertEqual(round(rounds * float(recipe['pp_warmup_min_ratio'])), 150)
        self.assertEqual(round(rounds * float(recipe['pp_gate_anneal_ratio'])), 75)
        self.assertAlmostEqual(cosine_learning_rate(300, .15, rounds), .05182372542187895)
        self.assertEqual(cosine_learning_rate(500, .15, rounds), .0001)

    def test_high_lr_changes_only_lr(self):
        base, experiment = config('reference'), config('high_lr')
        self.assertEqual({k for k in base if base[k] != experiment[k]}, {'lr_local_training'})
        self.assertEqual(experiment['lr_local_training'], '0.15')

    def test_trusted_changes_only_mode(self):
        base, experiment = config('reference'), config('trusted')
        self.assertEqual({k for k in base if base[k] != experiment[k]}, {'pp_geom_mode'})
        self.assertEqual(experiment['lr_local_training'], '0.1')
        self.assertEqual(experiment['pp_geom_mode'], 'trusted')

    def test_recipes_parse_as_supported_cli_arguments(self):
        for name in ('reference', 'high_lr', 'trusted', 'trusted_lr015',
                     'legacy_lr015_500', 'legacy_lr020'):
            argv = ['test']
            for key, value in config(name).items():
                argv.extend(['--' + key, value])
            with patch.object(sys, 'argv', argv):
                args = args_parser()
            self.assertEqual(args.max_rounds, 500 if name == 'legacy_lr015_500' else 300)
            self.assertEqual(args.lr_mid_factor, 1.)
            self.assertEqual(args.pp_b_conf_rescue, 0)
            self.assertEqual(args.num_workers, 16)
            self.assertEqual(args.seed, 7)

    def test_incompatible_trusted_settings_rejected(self):
        with patch.object(sys, 'argv', ['test', '--pp_geom_mode', 'trusted', '--pp_teacher', '1']):
            with self.assertRaises(SystemExit):
                args_parser()
