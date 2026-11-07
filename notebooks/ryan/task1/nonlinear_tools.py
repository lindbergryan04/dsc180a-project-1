"""Small-sample exploratory tools for the SAT3 notebook.

No source data are modified. Screening is descriptive; model assessment keeps
all isotope siblings of a letter together. Scalers and tuning fit training data.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, SplineTransformer, StandardScaler


def distance_correlation(x, y):
    """Biased sample distance correlation, with the conventional square root."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    # Rescaling avoids underflow for very small momentum fluxes.
    x, y = x / np.std(x), y / np.std(y)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    b = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    squared = np.mean(a * b) / np.sqrt(np.mean(a * a) * np.mean(b * b))
    return np.sqrt(np.clip(squared, 0, 1))


def linear_and_rank(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    return np.corrcoef(x, y)[0, 1], np.corrcoef(rankdata(x), rankdata(y))[0, 1]


def dependency_screen(frame, columns, groups, inputs):
    """All unordered pairs, plus sensitivity to deleting each letter group."""
    rows = []
    groups = np.asarray(groups)
    masks = [groups != label for label in np.unique(groups)]
    for a, b in combinations(columns, 2):
        x, y = frame[a].to_numpy(), frame[b].to_numpy()
        pearson, spearman = linear_and_rank(x, y)
        loo = []
        for keep in masks:
            r, rho = linear_and_rank(x[keep], y[keep])
            loo.append((r, rho, distance_correlation(x[keep], y[keep])))
        loo = np.array(loo)
        row = {
            'variable_1': a, 'variable_2': b,
            'kind': 'input-input' if a in inputs and b in inputs else (
                'output-output' if a not in inputs and b not in inputs else 'input-output'),
            'pearson': pearson, 'spearman': spearman,
            'distance_correlation': distance_correlation(x, y),
            'rank_minus_abs_linear': abs(spearman) - abs(pearson),
            'valid_deletions': int(np.isfinite(loo).all(axis=1).sum()),
        }
        for j, metric in enumerate(('pearson', 'spearman', 'distance')):
            row[f'{metric}_delete_min'] = np.nanmin(loo[:, j])
            row[f'{metric}_delete_max'] = np.nanmax(loo[:, j])
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_metrics(y, prediction, groups):
    """Each letter contributes total weight one, regardless of isotope count."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    groups = np.asarray(groups)
    count = pd.Series(groups).value_counts()
    weight = np.array([1 / count[label] for label in groups])
    return {
        'letter_MAE': np.average(np.abs(y - prediction), weights=weight),
        'letter_RMSE': np.sqrt(np.average((y - prediction) ** 2, weights=weight)),
        'letter_R2': r2_score(y, prediction, sample_weight=weight),
        'case_MAE': mean_absolute_error(y, prediction),
    }


def univariate_model(name):
    if name == 'linear':
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == 'quadratic':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
    if name == 'log_x':
        return make_pipeline(FunctionTransformer(np.log), StandardScaler(), Ridge(alpha=1.0))
    if name == 'inverse_x':
        return make_pipeline(FunctionTransformer(np.reciprocal), StandardScaler(), Ridge(alpha=1.0))
    if name == 'spline':
        return make_pipeline(SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation='linear'), StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(name)


def univariate_sweep(frame, features, targets, groups, progress=print):
    """Fixed-complexity 1D candidates. Selecting the best is exploratory."""
    groups = np.asarray(groups)
    folds = list(LeaveOneGroupOut().split(frame, groups=groups))
    rows, predictions = [], []
    for target in targets:
        y = frame[target].to_numpy()
        for feature in features:
            x = frame[[feature]].to_numpy(dtype=float)
            names = ['linear', 'quadratic', 'spline']
            if np.all(x > 0):
                names += ['log_x', 'inverse_x']
            for name in names:
                pred = np.empty(len(y))
                fallback = 0
                outside = np.zeros(len(y), dtype=bool)
                for train, test in folds:
                    levels = np.unique(x[train]).size
                    outside[test] = ((x[test] < x[train].min()) | (x[test] > x[train].max())).ravel()
                    if levels < 2:
                        pred[test] = y[train].mean()
                        fallback += 1
                        continue
                    candidate = name
                    if (name == 'quadratic' and levels < 3) or (name == 'spline' and levels < 4):
                        candidate = 'linear'
                        fallback += 1
                    model = univariate_model(candidate).fit(x[train], y[train])
                    pred[test] = model.predict(x[test])
                rows.append({
                    'target': target, 'input': feature, 'model': name,
                    **grouped_metrics(y, pred, groups),
                    'fallback_folds': fallback, 'extrapolation_cases': int(outside.sum()),
                })
                predictions.append(pd.DataFrame({
                    'case': frame.index, 'letter': groups, 'target': target,
                    'input': feature, 'model': name, 'actual': y, 'prediction': pred,
                    'outside_training_range': outside,
                }))
        progress(f'Univariate screen finished: {target}', flush=True)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def log_collision(x):
    """First column must be positive XNUE, as checked by the notebook."""
    result = np.array(x, dtype=float, copy=True)
    result[:, 0] = np.log10(result[:, 0])
    return result


def multivariate_candidates(positive_target):
    alphas = [0.1, 1.0, 10.0, 100.0]
    raw = make_pipeline(StandardScaler(), Ridge())
    logged = make_pipeline(FunctionTransformer(log_collision), StandardScaler(), Ridge())
    additive = make_pipeline(
        FunctionTransformer(log_collision),
        SplineTransformer(n_knots=3, degree=2, include_bias=False, extrapolation='linear'),
        StandardScaler(), Ridge(),
    )
    quadratic = make_pipeline(
        FunctionTransformer(log_collision), StandardScaler(),
        PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(),
    )
    result = {
        'mean': (DummyRegressor(strategy='mean'), {}),
        'linear': (raw, {'ridge__alpha': alphas}),
        'log_collision_linear': (logged, {'ridge__alpha': alphas}),
        'additive_splines': (additive, {'ridge__alpha': alphas}),
        'quadratic_interactions': (quadratic, {'ridge__alpha': alphas}),
        'random_forest': (RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=3,
            max_features=1.0, random_state=180, n_jobs=1), {}),
    }
    if positive_target:
        result['log_heat_linear'] = (
            TransformedTargetRegressor(regressor=clone(logged), func=np.log, inverse_func=np.exp),
            {'regressor__ridge__alpha': alphas},
        )
    return result


def nested_model_comparison(frame, features, targets, groups, progress=print):
    """Outer leave-one-letter-out; inner grouped CV chooses ridge penalty only."""
    if features[0] != 'XNUE' or not frame['XNUE'].gt(0).all():
        raise ValueError('The log-collision transform requires XNUE first and positive.')
    x, groups = frame[features].to_numpy(dtype=float), np.asarray(groups)
    folds = list(LeaveOneGroupOut().split(x, groups=groups))
    rows, predictions, tuning = [], [], []
    for target in targets:
        y = frame[target].to_numpy()
        for name, (estimator, params) in multivariate_candidates(target in ('Q_i', 'Q_e')).items():
            pred = np.empty(len(y))
            for train, test in folds:
                assert not set(groups[train]) & set(groups[test])
                if params:
                    inner = list(GroupKFold(5).split(x[train], y[train], groups[train]))
                    for inner_train, inner_test in inner:
                        assert not set(groups[train][inner_train]) & set(groups[train][inner_test])
                    fitted = GridSearchCV(
                        clone(estimator), params, cv=inner,
                        scoring='neg_mean_absolute_error', n_jobs=1, error_score='raise',
                    ).fit(x[train], y[train])
                    best = fitted.best_params_
                else:
                    fitted = clone(estimator).fit(x[train], y[train])
                    best = {}
                pred[test] = fitted.predict(x[test])
                tuning.append({'target': target, 'model': name, 'held_letter': groups[test][0], 'parameters': str(best)})
            if not np.isfinite(pred).all():
                raise ValueError(f'Non-finite held-out predictions: {target}, {name}')
            rows.append({
                'target': target, 'model': name, **grouped_metrics(y, pred, groups),
                'nonpositive_heat_predictions': int((pred <= 0).sum()) if target in ('Q_i', 'Q_e') else 0,
            })
            predictions.append(pd.DataFrame({
                'case': frame.index, 'letter': groups, 'target': target,
                'model': name, 'actual': y, 'prediction': pred,
            }))
            progress(f'Nested group validation finished: {target} / {name}', flush=True)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning)


def matched_rectangles(frame, features, targets):
    """Find every observed four-corner comparison with other inputs held fixed."""
    controls = frame[features].round(12)  # Remove roundoff from derived shear.
    results, coverage = [], []
    for first, second in combinations(features, 2):
        held = [name for name in features if name not in (first, second)]
        rectangle_count = 0
        for _, group in controls.groupby(held, sort=True):
            if group.duplicated([first, second]).any():
                raise ValueError('Repeated control coordinates need explicit replicate handling')
            cases = {(row[first], row[second]): case for case, row in group.iterrows()}
            for x0, x1 in combinations(sorted(group[first].unique()), 2):
                for z0, z1 in combinations(sorted(group[second].unique()), 2):
                    corners = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
                    if not all(corner in cases for corner in corners):
                        continue
                    rectangle_count += 1
                    ids = [cases[corner] for corner in corners]
                    for target in targets:
                        y00, y10, y01, y11 = frame.loc[ids, target].to_numpy()
                        contrast = y11 - y10 - y01 + y00
                        results.append({
                            'input_1': first, 'input_2': second, 'target': target,
                            'x_low': x0, 'x_high': x1, 'z_low': z0, 'z_high': z1,
                            'case_00': ids[0], 'case_10': ids[1], 'case_01': ids[2], 'case_11': ids[3],
                            'difference_of_differences': contrast,
                            'multiplicative_factor': (y11 * y00 / (y10 * y01)) if min(y00, y10, y01, y11) > 0 else np.nan,
                        })
        coverage.append({'input_1': first, 'input_2': second, 'matched_rectangles': rectangle_count})
    return pd.DataFrame(results), pd.DataFrame(coverage)


def baseline_scans(frame, features):
    baseline = frame.loc['a_D', features].astype(float)
    result = {}
    for feature in features:
        held = [name for name in features if name != feature]
        keep = np.isclose(frame[held].to_numpy(dtype=float), baseline[held].to_numpy(), rtol=1e-10, atol=1e-12).all(axis=1)
        result[feature] = frame.loc[keep].sort_values(feature)
    return result
