"""Identification of the Kallesoe reduced network model (eq. 8-13 of

    C. S. Kallesoe et al., "Smart Water Software ...", IFAC-PapersOnLine
    50-1 (2017) 6582-6587).

Fits the two Fourier demand models and the friction/head model from the
digitised paper data and writes every fitted parameter to data/model.json,
which kallesoe_mpc.py loads.

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

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(FOLDER, 'data', 'kallesoe_hourly.csv')
MODEL_JSON = os.path.join(FOLDER, 'data', 'model.json')

DEMAND_PERIOD_H = 24.0
LEVEL_PRESSURE_COEF = 0.1                            # alpha, eq. (2)
TOWER_HEIGHT = 30.0                                  # h0, eq. (2)
FIT_DAY_LO, FIT_DAY_HI = 2.0, 8.0


def load_digitised(path=DATA_CSV):
    columns = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key and value not in (None, ''):
                    columns.setdefault(key, []).append(float(value))
    return {key: np.array(values) for key, values in columns.items()}


def fourier_design_matrix(hours, harmonic_count):
    # basis of the truncated Fourier series g(t) of eq. (8)
    columns = [np.ones_like(hours)]
    for n in range(1, harmonic_count + 1):
        columns.append(np.cos(2 * np.pi * n * hours / DEMAND_PERIOD_H))
        columns.append(np.sin(2 * np.pi * n * hours / DEMAND_PERIOD_H))
    return np.column_stack(columns)


def friction_design_matrix(flow, demand_pz1):
    # basis of f_theta(d1, t) of eq. (12)
    abs_flow = np.sqrt(flow ** 2 + 1e-5)
    deviation = np.sqrt((flow - demand_pz1) ** 2 + 1e-5)
    return np.column_stack([abs_flow * flow, deviation * (flow - demand_pz1),
                            abs_flow * (flow - demand_pz1), deviation * flow,
                            np.ones_like(flow)])


def fit_models(harmonic_count):
    digitised = load_digitised()
    hour = digitised['hour']
    window = (hour >= 24 * FIT_DAY_LO) & (hour <= 24 * FIT_DAY_HI)
    hours = hour[window]
    design = fourier_design_matrix(hours, harmonic_count)

    model = {'period_h': DEMAND_PERIOD_H, 'n_harmonics': harmonic_count,
             'fit_window_days': [FIT_DAY_LO, FIT_DAY_HI]}

    # eq. (10): g_lambda <- least squares on the zone-1 demand
    # eq. (11): g_mu     <- least squares on the zone-2 demand
    for signal, key in [('water_demand1_m3h', 'g_lambda'),
                        ('water_demand2_m3h', 'g_mu')]:
        measured = digitised[signal][window]
        coefficients, *_ = np.linalg.lstsq(design, measured, rcond=None)
        model[key] = coefficients.tolist()
        print('%s: %d coefficients, fit RMSE %.3f m3/h (eq. 10/11)'
              % (key, len(coefficients),
                 float(np.sqrt(np.mean((design @ coefficients - measured) ** 2)))))

    # eq. (12)/(13): p1 - pn ~ f_theta(d1, t), linear in theta -> NNLS,
    # with pn = alpha*(h + h0) per eq. (2) at the digitised level
    flow = digitised['flow_d1_m3h'][window]
    demand_pz1 = digitised['water_demand1_m3h'][window]
    reservoir_pressure = LEVEL_PRESSURE_COEF * (digitised['level_m'][window] + TOWER_HEIGHT)
    target = digitised['pressure_p1_bar'][window] - reservoir_pressure
    theta, _ = nnls(friction_design_matrix(flow, demand_pz1), target)
    model['f_theta'] = dict(zip(['th1', 'th2', 'th3', 'th4', 'th5'],
                                [float(value) for value in theta]))
    fitted = friction_design_matrix(flow, demand_pz1) @ theta
    print('f_theta: %s, fit RMSE %.3f bar (eq. 13, NNLS)'
          % ({k: round(v, 6) for k, v in model['f_theta'].items()},
             float(np.sqrt(np.mean((fitted - target) ** 2)))))

    with open(MODEL_JSON, 'w') as f:
        json.dump(model, f, indent=2)
    print('wrote %s' % MODEL_JSON)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--harmonics', type=int, default=12,
                        help='Fourier terms k for the demand fits (default 12)')
    fit_models(parser.parse_args().harmonics)
