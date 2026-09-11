"""Lightweight checks for scalar YAML experiment isolation and CLI guards."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from options import args_parser


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
        for name in ('reference', 'high_lr', 'trusted'):
            argv = ['test']
            for key, value in config(name).items():
                argv.extend(['--' + key, value])
            with patch.object(sys, 'argv', argv):
                args = args_parser()
            self.assertEqual(args.max_rounds, 300)
            self.assertEqual(args.lr_mid_factor, 1.)
            self.assertEqual(args.pp_b_conf_rescue, 0)
            self.assertEqual(args.num_workers, 16)
            self.assertEqual(args.seed, 7)

    def test_incompatible_trusted_settings_rejected(self):
        with patch.object(sys, 'argv', ['test', '--pp_geom_mode', 'trusted', '--pp_teacher', '1']):
            with self.assertRaises(SystemExit):
                args_parser()
