"""Scientific and restart checks for the Task 5 search.

Run from the repository root with
``python -m unittest discover -s notebooks/ryan/task5 -p test_task5.py``.
Training fixtures use temporary directories and small synthetic data, never production runs.
The frozen-split test needs the Task 5 source data (``data/README.md``) and is skipped without it.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data
import engine
import search as experiment


def synthetic_frames():
    """Positive physical inputs/heat and signed particle flux with stable IDs."""
    rng = np.random.default_rng(180513)
    x = rng.uniform(.3, 2., size=(48, len(data.FEATURES)))
    x[:, -1] = np.tile([.2, .4, .6, .8], 12)
    frame = pd.DataFrame(x, columns=data.FEATURES)
    frame['Q_e'] = np.exp(.3 * x[:, 0] + .2 * x[:, 4])
    frame['Q_i'] = np.exp(.5 * x[:, 1] - .2 * x[:, 2])
    frame['Gamma_e'] = x[:, 2] - 1.
    frame['sample_id'] = [f'synthetic_{i:03d}' for i in range(len(frame))]
    train, validation = frame.iloc[:32].copy(), frame.iloc[32:].copy()
    train.attrs['split'] = 'train'
    validation.attrs['split'] = 'validation'
    return train, validation


def tiny_config():
    row = pd.read_csv(ROOT / 'search_matrix.csv', keep_default_na=False).iloc[0].to_dict()
    row.update(case_id='synthetic_joint', width=4, depth=1, bottleneck_width=2,
               warmup_epochs=2, main_epochs=3, batch_size=8, patience_epochs=20,
               eval_every_epochs=1, parameters_per_member=47)
    return row


@unittest.skipUnless(all(path.is_file() for path in data.SOURCE_PATHS.values()),
                     'Task 5 source data are not present (see data/README.md)')
class FrozenDataTests(unittest.TestCase):
    def test_approved_membership_and_signed_particles(self):
        with tempfile.TemporaryDirectory(prefix='tglf-split-check-') as directory:
            root = Path(directory)
            manifest = data.prepare_data(root)
            self.assertEqual(manifest['membership_sha256'], data.MEMBERSHIP_SHA256)
            train, validation = data.load_split('train', root), data.load_split('validation', root)
            self.assertEqual((len(train), len(validation), manifest['test_rows']), (5131, 1100, 1100))
            self.assertFalse(set(train.sample_id) & set(validation.sample_id))
            self.assertTrue(train.Gamma_e.lt(0).any())
            self.assertFalse(train[['Q_e', 'Q_i']].lt(0).any().any())
            membership = pd.read_csv(root / 'data' / 'split_assignments.csv')
            self.assertFalse(membership.sample_id.duplicated().any())
            self.assertTrue(membership.groupby('RMIN_LOC')['split'].nunique().eq(3).all())
            # Rebuilding from the source tables reproduces the committed split byte for byte.
            self.assertEqual((root / 'data' / 'split_assignments.csv').read_bytes(),
                             (ROOT / 'split.csv').read_bytes())


class PreprocessingTests(unittest.TestCase):
    def test_roundtrips_preserve_positive_and_signed_values(self):
        for kind in sorted(engine.TRANSFORMS):
            values = ([.001, .1, 1., 3., 100.] if kind in {'log10', 'log1p'}
                      else [-100., -3., -.01, 0., .2, 10., 1000.])
            frame = pd.DataFrame({'flux': values})
            with self.subTest(kind=kind):
                transform = engine.Preprocessor.fit(frame, ['flux'], [kind])
                actual = transform.inverse_transform(transform.transform(frame))[:, 0]
                np.testing.assert_allclose(actual, values, rtol=2e-10, atol=2e-10)

    def test_validation_values_do_not_change_fitted_parameters(self):
        train = pd.DataFrame({'flux': [1., 2., 4., 8.]})
        transform = engine.Preprocessor.fit(train, ['flux'], ['identity'])
        before = copy.deepcopy(transform.state)
        validation = pd.DataFrame({'flux': [100., 1000.]})
        standardized = transform.transform(validation)[:, 0]
        np.testing.assert_allclose(standardized, (validation.flux - train.flux.mean()) / train.flux.std(ddof=0))
        self.assertEqual(transform.state, before)
        self.assertEqual(transform.state['n_training_cases'], 4)

    def test_log_rejects_zero_and_negative_values(self):
        for bad in [0., -1.]:
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, 'positive'):
                engine.Preprocessor.fit(pd.DataFrame({'flux': [1., bad, 3.]}), ['flux'], ['log10'])

    def test_inverse_overflow_and_yeo_johnson_domain_fail_explicitly(self):
        transform = engine.Preprocessor.fit(pd.DataFrame({'flux': [1., 10., 100.]}), ['flux'], ['log10'])
        with self.assertRaises(ValueError):
            transform.inverse_transform(np.array([[1e9]]))
        for power, invalid in [(-1., 2.), (3., -2.)]:
            transform = engine.Preprocessor({'columns': [{'name': 'flux', 'kind': 'yeo_johnson',
                        'scale': 1., 'power': power}], 'mean': [0.], 'scale': [1.]})
            with self.subTest(power=power), self.assertRaisesRegex(ValueError, 'domain'):
                transform.inverse_transform(np.array([[invalid]]))


class TrainingRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='tglf-engine-check-')
        cls.base = Path(cls.directory.name)
        cls.train, cls.validation = synthetic_frames()
        cls.config = tiny_config()
        cls.complete = cls.base / 'continuous'
        cls.resumed = cls.base / 'resumed'
        cls.summary = engine.run_case(cls.config, cls.train, cls.validation, cls.complete, threads=1)
        cls.first_pause = engine.run_case(cls.config, cls.train, cls.validation, cls.resumed,
                                          threads=1, stop_after_epochs=1)
        cls.paused_has_metrics = (cls.resumed / 'metrics.csv').exists()
        try:
            engine.load_ensemble(cls.resumed)
            cls.incomplete_rejected = False
        except ValueError:
            cls.incomplete_rejected = True
        cls.second_pause = engine.run_case(cls.config, cls.train, cls.validation, cls.resumed,
                                           threads=1, stop_after_epochs=3)
        cls.paused_bundle = torch.load(cls.resumed / 'checkpoints.pt', weights_only=True)
        cls.resumed_summary = engine.run_case(cls.config, cls.train, cls.validation, cls.resumed, threads=1)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_pause_crosses_warmup_and_main_without_scoring_incomplete_ensemble(self):
        self.assertEqual(self.first_pause['status'], 'paused')
        self.assertEqual(self.second_pause['status'], 'paused')
        self.assertFalse(self.paused_has_metrics)
        self.assertTrue(self.incomplete_rejected)
        self.assertEqual(self.paused_bundle['members'][0]['phase'], 'main')
        self.assertEqual(self.paused_bundle['members'][0]['epoch'], 2)

    def test_restart_matches_continuous_weights_predictions_and_history(self):
        self.assertEqual(self.resumed_summary['status'], 'complete')
        full = torch.load(self.complete / 'checkpoints.pt', weights_only=True)
        resumed = torch.load(self.resumed / 'checkpoints.pt', weights_only=True)
        self.assertEqual(len(full['members']), 5)
        for a, b in zip(full['members'], resumed['members']):
            self.assertEqual(a['best_epoch'], b['best_epoch'])
            self.assertEqual(a['history'], b['history'])
            self.assertTrue(torch.equal(a['generator_state'], b['generator_state']))
            for name in a['best_weights']:
                self.assertTrue(torch.equal(a['best_weights'][name], b['best_weights'][name]), name)
        left = engine.load_ensemble(self.complete).predict(self.validation)
        right = engine.load_ensemble(self.resumed).predict(self.validation)
        np.testing.assert_array_equal(left['members'], right['members'])
        np.testing.assert_array_equal(left['prediction'], right['prediction'])
        pd.testing.assert_frame_equal(pd.read_csv(self.complete / 'history.csv'),
                                      pd.read_csv(self.resumed / 'history.csv'))

    def test_fitted_preprocessing_contains_only_training_information(self):
        bundle = torch.load(self.complete / 'checkpoints.pt', weights_only=True)
        xstate, ystate = bundle['input_preprocessing'], bundle['target_preprocessing']
        self.assertEqual(xstate['n_training_cases'], len(self.train))
        self.assertEqual(ystate['n_training_cases'], len(self.train))
        self.assertAlmostEqual(xstate['mean'][4], np.log10(self.train.XNUE).mean())
        self.assertAlmostEqual(ystate['mean'][2], self.train.Gamma_e.mean())
        predictions = pd.read_csv(self.complete / 'predictions.csv')
        self.assertEqual(set(predictions['split']), {'validation'})
        self.assertEqual(set(predictions.sample_id), set(self.validation.sample_id))

    def test_completed_run_replays_and_rejects_changed_config_or_data(self):
        before = (self.complete / 'checkpoints.pt').read_bytes()
        outcome = engine.run_case(self.config, self.train, self.validation, self.complete, threads=1)
        self.assertEqual(outcome['status'], 'complete')
        self.assertEqual(before, (self.complete / 'checkpoints.pt').read_bytes())
        changed_config = dict(self.config, width=8)
        with self.assertRaisesRegex(ValueError, 'identity'):
            engine.run_case(changed_config, self.train, self.validation, self.complete, threads=1)
        changed_data = self.train.copy()
        changed_data.loc[changed_data.index[0], 'Q_e'] += .2
        with self.assertRaisesRegex(ValueError, 'identity'):
            engine.run_case(self.config, changed_data, self.validation, self.complete, threads=1)

    def test_average_is_taken_after_each_inverse_transform(self):
        ensemble = engine.load_ensemble(self.complete)
        standardized = np.array([[.2*i, .1*i, .05*i] for i in range(5)])
        with torch.no_grad():
            for model, output in zip(ensemble.models, standardized):
                for parameter in model.parameters():
                    parameter.zero_()
                model[-1].bias.copy_(torch.as_tensor(output, dtype=torch.float32))
        predictions = ensemble.predict(self.validation.iloc[:2])
        member_truth = ensemble.yprocessor.inverse_transform(standardized.astype(np.float32))
        np.testing.assert_allclose(predictions['prediction'][0], member_truth.mean(axis=0), rtol=1e-12)
        wrong_order = ensemble.yprocessor.inverse_transform(standardized.mean(axis=0, keepdims=True))[0]
        self.assertGreater(abs(predictions['prediction'][0, 0] - wrong_order[0]), 1e-4)


def mock_success(root, row, score=1., repeat=0):
    """Minimal saved observations for selection tests; no networks are claimed trained."""
    folder = root / 'runs' / row['case_id'] / f'repeat_{repeat}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'summary.json').write_text(json.dumps({'status': 'complete'}))
    targets = data.TARGETS if row['output_mode'] == 'joint' else [row['output_mode']]
    pd.DataFrame([{'case_id': row['case_id'], 'repeat': repeat, 'member_seed': 'ensemble',
                   'split': 'validation', 'target': target, 'rmse': score, 'nrmse': score}
                  for target in targets]).to_csv(folder / 'metrics.csv', index=False)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='tglf-selection-check-')
        self.root = Path(self.directory.name)
        (self.root / 'experiment_matrix.csv').write_bytes((ROOT / 'search_matrix.csv').read_bytes())
        self.matrix = experiment.validate_matrix(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def _complete_screening(self):
        for row in self.matrix.to_dict('records'):
            # Reference beats every nonreference, testing reference retention.
            mock_success(self.root, row, score=1. if row['phase'] == 'control' else 2.)

    def test_matrix_enforces_declared_single_change(self):
        self.assertEqual(len(self.matrix), 114)
        modified = self.matrix.copy()
        index = modified.index[modified.case_id.eq('joint_width_128')][0]
        modified.loc[index, 'hidden_activation'] = 'relu'
        modified.to_csv(self.root / 'experiment_matrix.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Undeclared changes'):
            experiment.validate_matrix(self.root)

    def test_test_labels_remain_unopened_before_final_selection(self):
        with mock.patch.object(data, 'load_split', side_effect=AssertionError('Loaded held-out data')) as load, \
             mock.patch.object(data, 'load_profile_test', side_effect=AssertionError('Loaded profile data')) as profile:
            with self.assertRaisesRegex(RuntimeError, 'preceding stage'):
                experiment.evaluate_final(self.root)
            load.assert_not_called()
            profile.assert_not_called()

    def test_selected_factorial_and_confirmation_keep_controls(self):
        self._complete_screening()
        interactions = experiment.build_interactions(self.root)
        self.assertEqual(len(interactions), 44)
        selection = json.loads((self.root / 'results' / 'selection.json').read_text())
        self.assertEqual(len(selection['interactions']['grid']), 64)
        for parents in selection['interactions']['parents'].values():
            self.assertEqual(set(parents), {'structure', 'activation', 'input', 'target'})
            self.assertEqual(len(set(parents.values())), 4)
        # Observations changing after the freeze must not redefine the interaction grid.
        changed = self.matrix.loc[self.matrix.case_id.eq('joint_width_256')].iloc[0].to_dict()
        mock_success(self.root, changed, score=99.)
        pd.testing.assert_frame_equal(experiment.build_interactions(self.root), interactions)
        for row in interactions.to_dict('records'):
            mock_success(self.root, row, score=3.)
        confirmation = experiment.build_confirmation(self.root)
        self.assertLessEqual(len(confirmation), 12)
        self.assertTrue(confirmation.main_epochs.eq(600).all())
        self.assertTrue({'joint_base', 'qe_base', 'qi_base', 'gamma_base'} <= set(confirmation.source_case_id))
        self.assertEqual(experiment.REPEATS, {1:'142|143|144|145|146',
                                            2:'242|243|244|245|246', 3:'342|343|344|345|346'})
        for row in confirmation.to_dict('records'):
            for repeat in (1, 2, 3):
                mock_success(self.root, row, score=1. if row['source_case_id'].endswith('_base') else 2., repeat=repeat)
        final = experiment.freeze_final_selection(self.root)
        self.assertEqual(final['selection_split'], 'validation')
        self.assertFalse(final['test_opened'])
        self.assertEqual(set(final['systems']['shared'].values()), {'confirm_joint_base'})
        self.assertEqual(final['systems']['separate'], {'Q_e':'confirm_qe_base',
                                                      'Q_i':'confirm_qi_base', 'Gamma_e':'confirm_gamma_base'})


class FinalEvaluationTests(unittest.TestCase):
    def test_real_checkpoint_evaluation_and_selected_deployment(self):
        """Exercise real inference/CSV outputs behind a mocked selection boundary."""
        with tempfile.TemporaryDirectory(prefix='tglf-final-check-') as directory:
            root = Path(directory).resolve()
            (root / 'experiment_matrix.csv').write_bytes((ROOT / 'search_matrix.csv').read_bytes())
            train, validation = synthetic_frames()
            config = dict(tiny_config(), warmup_epochs=1, main_epochs=1)
            for repeat, seeds in experiment.REPEATS.items():
                repeated_config = dict(config, repeat=repeat, member_seeds=seeds)
                engine.run_case(repeated_config, train, validation,
                                root / 'runs' / config['case_id'] / f'repeat_{repeat}', threads=1)

            test = validation.iloc[:8].copy()
            test['sample_id'] = 'heldout_' + test.sample_id
            profile = validation.iloc[8:].copy()
            profile['sample_id'] = 'profile_' + profile.sample_id
            profile['RMIN_LOC'] += .025
            system_names = ['shared', 'output_specific_shared', 'separate',
                            'shared_reference', 'separate_reference', 'capacity_control']
            # Every system deliberately reuses one real joint model. This tests deduplication
            # of model calls and cost accounting, without pretending these are tuned winners.
            final = {'systems': {name: {target: config['case_id'] for target in data.TARGETS}
                                  for name in system_names},
                     'evaluated_case_ids': [config['case_id']],
                     'selection_split': 'validation', 'test_opened': False}
            selection = experiment._selection(root)
            selection['final'] = final
            experiment._save_selection(root, selection)

            with mock.patch.object(experiment, 'freeze_final_selection', side_effect=RuntimeError('not frozen')), \
                 mock.patch.object(data, 'load_split') as split_loader, \
                 mock.patch.object(data, 'load_profile_test') as profile_loader:
                with self.assertRaisesRegex(RuntimeError, 'not frozen'):
                    experiment.evaluate_final(root, threads=1)
                split_loader.assert_not_called()
                profile_loader.assert_not_called()

            with mock.patch.object(experiment, 'freeze_final_selection', return_value=final), \
                 mock.patch.object(data, 'load_split', return_value=test) as split_loader, \
                 mock.patch.object(data, 'load_profile_test', return_value=profile) as profile_loader:
                actual = experiment.evaluate_final(root, threads=1)
                split_loader.assert_called_once_with('test', root)
                profile_loader.assert_called_once_with(root)

            self.assertTrue(actual['test_opened'])
            self.assertEqual(actual['evaluated_datasets'], {'test': 8, 'profile_test': 8})
            costs = pd.read_csv(root / 'results' / 'system_costs.csv')
            self.assertEqual(len(costs), len(system_names) * 3 * 2)
            self.assertTrue(costs.unique_ensembles.eq(1).all())
            self.assertTrue(costs.networks.eq(5).all())
            self.assertTrue(costs.parameters_total.eq(5 * engine.parameter_count(config)).all())
            self.assertTrue(np.isfinite(costs[['inference_single_seconds', 'inference_batch_seconds']]).all().all())
            metrics = pd.read_csv(root / 'results' / 'system_metrics.csv')
            self.assertEqual(set(metrics['split']), {'test', 'profile_test'})
            self.assertEqual(set(metrics.target), set(data.TARGETS))
            self.assertEqual(len(metrics), len(system_names) * 3 * 2 * 3)
            self.assertTrue(np.isfinite(metrics[['rmse', 'nrmse', 'mae', 'bias']]).all().all())
            self.assertTrue((root / 'results' / 'system_radius_metrics.csv').is_file())
            self.assertTrue((root / 'results' / 'system_predictions.csv').is_file())
            run_predictions = pd.read_csv(root / 'runs' / config['case_id'] / 'repeat_1' / 'predictions.csv')
            self.assertEqual(set(run_predictions['split']), {'validation', 'test', 'profile_test'})

            deployment = experiment.load_selected_system(root=root, repeat=1)
            self.assertEqual(len(deployment.ensembles), 1)
            inputs = test[data.FEATURES + ['sample_id']].copy()
            inputs.index = [f'row_{i}' for i in range(len(inputs))]
            only_ensemble = next(iter(deployment.ensembles.values()))
            with mock.patch.object(only_ensemble, 'predict', wraps=only_ensemble.predict) as predict:
                deployed = deployment.predict(inputs)
                predict.assert_called_once()
            self.assertEqual(list(deployed.columns), data.TARGETS)
            self.assertEqual(list(deployed.index), list(inputs.index))
            expected = engine.load_ensemble(root / 'runs' / config['case_id'] / 'repeat_1').predict(inputs)['prediction']
            np.testing.assert_array_equal(deployed.to_numpy(), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
