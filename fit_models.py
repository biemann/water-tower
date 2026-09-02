"""Identification of the Kallesoe reduced network model (eq. 8-13 of

    C. S. Kallesoe et al., "Smart Water Software ...", IFAC-PapersOnLine
    50-1 (2017) 6582-6587).

Fits the two Fourier demand models and the friction/head model from the
digitised paper data (kallesoe_hourly.csv) and writes every fitted
parameter to data/model.json, which kallesoe_receding_horizon.py loads.

    python fit_models.py                 # 12 harmonics (default)
    python fit_models.py --harmonics 8   # fewer Fourier terms

Requires: numpy, scipy.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.optimize import nnls

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(_HERE, 'data', 'kallesoe_hourly.csv')
MODEL_JSON = os.path.join(_HERE, 'data', 'model.json')

PERIOD_H = 24.0
ALPHA = 0.1
H0_TWR = 30.0
FIT_DAY_LO, FIT_DAY_HI = 2.0, 8.0   # model-valid region, days 2-8


def load_digitised(path=DATA_CSV):
    cols = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key and val not in (None, ''):
                    cols.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in cols.items()}


def fourier_matrix(t_h, n_harm):
    # basis of the truncated Fourier series g(t) of eq. (8)
    cols = [np.ones_like(t_h)]
    for n in range(1, n_harm + 1):
        cols.append(np.cos(2 * np.pi * n * t_h / PERIOD_H))
        cols.append(np.sin(2 * np.pi * n * t_h / PERIOD_H))
    return np.column_stack(cols)


def fit_models(n_harm):
    fig = load_digitised()
    t = fig['hour']
    win = (t >= 24 * FIT_DAY_LO) & (t <= 24 * FIT_DAY_HI)
    th = t[win]
    X = fourier_matrix(th, n_harm)

    model = {'period_h': PERIOD_H, 'n_harmonics': n_harm,
             'fit_window_days': [FIT_DAY_LO, FIT_DAY_HI]}

    # eq. (10): g_lambda <- least squares on the PZ1 demand dbar
    # eq. (11): g_mu     <- least squares on the PZ2 demand dn+1
    for sig, key in [('flow_dbar_m3h', 'g_lambda'),
                     ('flow_dn1_m3h', 'g_mu')]:
        y = fig[sig][win]
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        model[key] = c.tolist()
        print('%s: %d coefficients, fit RMSE %.3f m3/h (eq. 10/11)'
              % (key, len(c), float(np.sqrt(np.mean((X @ c - y) ** 2)))))

    # eq. (12)/(13): p1 - p_n ~ f_theta(d1, t), linear in theta -> NNLS.
    # p_n = alpha*(h + h0) per eq. (2), evaluated with the digitised level.
    dbar = fig['flow_dbar_m3h'][win]
    d1 = fig['flow_d1_m3h'][win]
    target = fig['pressure_p1_bar'][win] - ALPHA * (fig['level_m'][win] + H0_TWR)
    sus = np.sqrt(d1 ** 2 + 1e-9)
    diff = np.sqrt((d1 - dbar) ** 2 + 1e-9)
    basis = np.column_stack([sus * d1, diff * (d1 - dbar),
                             sus * (d1 - dbar), diff * d1, np.ones_like(d1)])
    theta, _ = nnls(basis, target)
    model['f_theta'] = dict(zip(['th1', 'th2', 'th3', 'th4', 'th5'],
                                [float(v) for v in theta]))
    fit = basis @ theta
    print('f_theta: %s, fit RMSE %.3f bar (eq. 13, NNLS)'
          % ({k: round(v, 6) for k, v in model['f_theta'].items()},
             float(np.sqrt(np.mean((fit - target) ** 2)))))

    with open(MODEL_JSON, 'w') as f:
        json.dump(model, f, indent=2)
    print('wrote %s' % MODEL_JSON)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--harmonics', type=int, default=12,
                    help='Fourier terms k for the demand fits (default 12)')
    fit_models(ap.parse_args().harmonics)
