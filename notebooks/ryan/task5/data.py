"""Task 5 data: combine both teams' DIII-D tables, rebuild the frozen split, read the profile test.

* Team A is our Task 4 table, ``notebooks/ryan/task4/d3d_team_database.csv``.
* Team B (``team_b_d3d_database.csv``) and the 12 intermediate-radius profile cases
  (``d3d_profile_test.tar.gz``) are course files in ``data/Task5_DIIID_Surrogate/``.
  Set ``TASK5_DATA_DIR`` to read them from elsewhere.

``root`` is an experiment directory (default ``notebooks/ryan/runs/task5/search``) that
receives ``data/split_assignments.csv``. The split must match ``split.csv`` in this folder
byte for byte (checked through its SHA-256).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[2]
ROOT = REPO / 'notebooks/ryan/runs/task5/search'
DATA_DIR = Path(os.environ.get('TASK5_DATA_DIR', REPO / 'data/Task5_DIIID_Surrogate')).resolve()
FEATURES = ['RLTS_1', 'RLTS_2', 'RLNS_1', 'TAUS_2', 'XNUE', 'RMIN_LOC']
TARGETS = ['Q_e', 'Q_i', 'Gamma_e']
SOURCES = {
    'team_a_d3d_database.csv': '1439a46a93c35b0dfe59de504b7a04a50dcf01b32562e99d984856f8e4c13653',
    'team_b_d3d_database.csv': '32be24624677181e7d41c92ee6cc9e85454a58180d09465183f81bf4118e7864',
}
MEMBERSHIP_SHA256 = '6b82be348dd5176b4342bec0aa3c196dc3295c23356237b59373df577c8a407f'
SOURCE_PATHS = {
    'team_a_d3d_database.csv': REPO / 'notebooks/ryan/task4/d3d_team_database.csv',
    'team_b_d3d_database.csv': DATA_DIR / 'team_b_d3d_database.csv',
}
PROFILE_TEST = DATA_DIR / 'd3d_profile_test.tar.gz'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _root(root=None):
    return Path(root).resolve() if root is not None else ROOT


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def _clean(root):
    tables = []
    for name, expected in SOURCES.items():
        path = SOURCE_PATHS[name]
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. See data/README.md for the Task 5 inputs.')
        if sha256(path) != expected:
            raise ValueError(f'{name} changed since the approved split. Use a new experiment version.')
        frame = pd.read_csv(path)
        aliases = {'member': 'contributor'}
        for quantity in ['RLTS', 'RLNS', 'TAUS', 'AS', 'MASS', 'ZS']:
            aliases.update({f'{quantity}_e': f'{quantity}_1', f'{quantity}_i': f'{quantity}_2'})
        frame = frame.rename(columns=aliases)
        if frame.columns.duplicated().any():
            raise ValueError(f'Conflicting aliases in {name}')
        missing = set(FEATURES + TARGETS + ['contributor', 'case']) - set(frame.columns)
        if missing:
            raise ValueError(f'{name} is missing {sorted(missing)}')
        if not (frame.ZS_1.eq(-1).all() and frame.ZS_2.eq(1).all()
                and frame.MASS_1.lt(frame.MASS_2).all()):
            raise ValueError(f'Unexpected species mapping in {name}')
        frame['source_csv'] = name
        frame['source_row'] = np.arange(2, len(frame) + 2)
        # IDs keep the original file stem so the frozen membership hash still applies.
        frame['sample_id'] = Path(name).stem + ':' + frame.contributor.astype(str) + ':' + frame.case.astype(str)
        frame[FEATURES + TARGETS] = frame[FEATURES + TARGETS].apply(pd.to_numeric, errors='coerce')
        tables.append(frame)
    raw = pd.concat(tables, ignore_index=True)
    if raw.sample_id.duplicated().any():
        raise ValueError('Nonunique source IDs')
    finite = np.isfinite(raw[FEATURES + TARGETS]).all(axis=1)
    positive = raw.XNUE.gt(0) & raw.TAUS_2.gt(0)
    negative_heat = raw[['Q_e', 'Q_i']].lt(0).any(axis=1)
    reasons = pd.Series('', index=raw.index)
    reasons.loc[~finite] = 'nonfinite_required_value'
    reasons.loc[finite & ~positive] = 'nonpositive_XNUE_or_TAUS_2'
    reasons.loc[finite & positive & negative_heat] = 'negative_heat_flux'
    eligible = raw.loc[reasons.eq('')]
    fixed = ['NS', 'NBASIS_MAX', 'NBASIS_MIN', 'KAPPA_LOC', 'Q_LOC',
             'Q_PRIME_LOC', 'DRMAJDX_LOC', 'RMAJ_LOC', 'S_KAPPA_LOC', 'DELTA_LOC', 'S_DELTA_LOC']
    if eligible[fixed].isna().any().any() or eligible.groupby('RMIN_LOC')[fixed].nunique().gt(1).any().any():
        raise ValueError('Fixed simulation fields differ or are missing at a radius')
    if not np.allclose(eligible.RLNS_1, eligible.RLNS_2):
        raise ValueError('Electron/ion density gradients differ')
    radii = np.sort(eligible.RMIN_LOC.unique())
    if len(radii) != 13 or not np.allclose(radii, np.arange(20, 81, 5) / 100):
        raise ValueError('Expected the original 13 radii')
    repeated = eligible.loc[eligible.duplicated(FEATURES, keep=False)]
    if repeated.groupby(FEATURES)[TARGETS].nunique().gt(1).any().any():
        raise ValueError('Repeated inputs have conflicting targets')
    reasons.loc[eligible.index[eligible.duplicated(FEATURES)]] = 'duplicate_features'
    clean = raw.loc[reasons.eq('')].sort_values('sample_id').reset_index(drop=True)
    return clean, raw, reasons


def _membership(clean):
    train_ids, holdout_ids = train_test_split(clean.index.to_numpy(), test_size=.30,
        random_state=180510, stratify=clean.RMIN_LOC)
    validation_ids, test_ids = train_test_split(holdout_ids, test_size=.50,
        random_state=180511, stratify=clean.loc[holdout_ids, 'RMIN_LOC'])
    membership = clean[['sample_id', 'source_csv', 'contributor', 'RMIN_LOC']].copy()
    membership['split'] = 'train'
    membership.loc[validation_ids, 'split'] = 'validation'
    membership.loc[test_ids, 'split'] = 'test'
    serialized = membership.to_csv(index=False, lineterminator='\n')
    if hashlib.sha256(serialized.encode()).hexdigest() != MEMBERSHIP_SHA256:
        raise ValueError('Split differs from the approved notebook. Check package versions before proceeding.')
    return membership, serialized


def prepare_data(root=None):
    """Freeze verified membership without exporting duplicate copies of the data."""
    root = _root(root)
    clean, raw, reasons = _clean(root)
    membership, serialized = _membership(clean)
    if membership['split'].value_counts().to_dict() != {'train': 5131, 'validation': 1100, 'test': 1100}:
        raise ValueError('Unexpected split counts')
    path = root / 'data' / 'split_assignments.csv'
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != MEMBERSHIP_SHA256:
            raise ValueError('Existing split file differs; refusing to replace it')
    else:
        _atomic_text(path, serialized)
    manifest = {
        'source_sha256': SOURCES, 'membership_sha256': MEMBERSHIP_SHA256,
        'features': FEATURES, 'targets': TARGETS, 'input_rows': len(raw),
        'excluded_rows': int(reasons.ne('').sum()),
        'exclusions': reasons.loc[reasons.ne('')].value_counts().to_dict(),
        'training_rows': 5131, 'validation_rows': 1100, 'test_rows': 1100,
        'split_seed': 180510, 'validation_test_split_seed': 180511,
        'fractions': {'train': .70, 'validation': .15, 'test': .15},
        'stratified_by': 'RMIN_LOC',
        'signed_particle_flux': 'retained',
        'team_b_missing_fields': ['SAT_RULE', 'KYGRID_MODEL', 'NXGRID'],
        'team_b_assumption': 'Task 4 simulation recipe',
        'evaluation_history': 'Earlier quick demonstration used this cohort under another split. Test reserved from this experiment onward.',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
    }
    target = root / 'data' / 'dataset_manifest.json'
    if target.exists():
        existing = json.loads(target.read_text())
        for key in ['source_sha256', 'membership_sha256', 'features', 'targets']:
            if existing[key] != manifest[key]:
                raise ValueError(f'Dataset manifest changed: {key}')
        return existing
    _atomic_text(target, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def load_split(split, root=None):
    """Load one split. The runner requests test only after freezing model selection."""
    if split not in {'train', 'validation', 'test'}:
        raise ValueError(f'Unknown split {split}')
    root = _root(root)
    manifest = prepare_data(root)
    clean, _, _ = _clean(root)
    membership = pd.read_csv(root / 'data' / 'split_assignments.csv')
    ids = membership.loc[membership['split'].eq(split), 'sample_id']
    frame = clean.loc[clean.sample_id.isin(ids)].sort_values('sample_id').reset_index(drop=True)
    frame.attrs['dataset_identity'] = {k: manifest[k] for k in ['source_sha256', 'membership_sha256']}
    frame.attrs['split'] = split
    return frame


def load_profile_test(root=None):
    """Read the 12 supplied new-radius inputs and labels directly from their archive."""
    root = _root(root)
    path = PROFILE_TEST
    inputs, answers = [], []
    with tarfile.open(path, 'r:gz') as archive:
        for member in sorted(archive.getmembers(), key=lambda x: x.name):
            if member.isfile() and member.name.endswith('/tglf_sat2/input.tglf'):
                values = {}
                for line in archive.extractfile(member).read().decode().splitlines():
                    line = line.split('#', 1)[0].split('!', 1)[0]
                    if '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    try:
                        values[key.strip()] = float(value.strip().replace('D', 'E').replace('d', 'e'))
                    except ValueError:
                        pass
                inputs.append({'sample_id': Path(member.name).parts[-3], **values})
        text = archive.extractfile('d3d_profile_test/d3d_profile_test_answers.dat').read().decode()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[0].startswith('r'):
            answers.append([fields[0], *map(float, fields[1:])])
    labels = pd.DataFrame(answers, columns=['sample_id', 'answer_radius', *TARGETS])
    frame = pd.DataFrame(inputs).merge(labels, on='sample_id', validate='one_to_one')
    if len(frame) != 12 or not np.allclose(frame.RMIN_LOC, frame.answer_radius):
        raise ValueError('Expected 12 aligned profile inputs and answers')
    if not np.isfinite(frame[FEATURES + TARGETS]).all().all():
        raise ValueError('Nonfinite profile-test values')
    if not np.allclose(np.sort(frame.RMIN_LOC), np.arange(225, 776, 50) / 1000):
        raise ValueError('Unexpected profile-test radii')
    frame.attrs['split'] = 'profile_test'
    frame.attrs['source_sha256'] = sha256(path)
    return frame.sort_values('sample_id').reset_index(drop=True)
