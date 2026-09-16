"""Predictive control of the Kallesoe water-tower network.

Implementation of the water-tower case study (v1) of

    Kallesøe, Carsten Skovmose, Tom Nørgaard Jensen, and Jan Dimon Bendtsen.
    "Plug-and-play model predictive control for water supply networks with
    storage." IFAC-PapersOnLine 50.1 (2017): 6582-6587.

By default this runs the open-loop economic MPC over the full 10-day window
(v1: deterministic, one solve) and plots the result.  Use --receding for the
closed-loop experiment, which follows the v2 architecture (Kallesøe et al.,
IFAC-PapersOnLine 56-2, 2023): 24 h receding horizon with chance-constrained
level bounds, mean-scaled stochastic demand and a simplified local safety controller (v2 Sec. 3.2) projecting the
pump flow so the level bounds hold.  Equation numbers without a version refer
to the v1 paper; the model is identified by fit_models.py and read from
data/model.json.

    python kallesoe_mpc.py               # open loop + figure
    python kallesoe_mpc.py --receding    # closed loop, 3 seeds
    python kallesoe_mpc.py --no-plot     # skip the figure

Requires: numpy, casadi (IPOPT), matplotlib (figure).
"""
import argparse
import csv
import json
import os

import numpy as np
import casadi as ca

from kallesoe_plot import trajectory

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(FOLDER, 'data', 'kallesoe_hourly.csv')
MODEL_JSON = os.path.join(FOLDER, 'data', 'model.json')

WATER_LEVEL_MIN, WATER_LEVEL_MAX = 1.0, 2.0           # bounds of eq. (14d)
TIME_STEP = 1.0
AREA = 400.0                                          # lambda0 = 1/area, eq. (9)
LEVEL_PRESSURE_COEF = 0.1                             # alpha, eq. (2)
TOWER_HEIGHT = 30.0                                   # h0, eq. (2)
PRESSURE_BEFORE_PUMP = 2.408049                       # p0, eq. (14a)
KAPPA = 0.062893                                      # terminal weight, eq. (14a)
KAPPA_RAMP = 0.0                                      # v2 (13a) control-ramp weight
UNIT_CONVERSION = 0.02778                             # bar*m3/h -> kW
PUMP_EFFICIENCY = 0.65                                # eta, eq. (14a)
WATER_QUALITY = 100.0                                 # water-quality exchange threshold of (17), (23c)

DEMAND_PERIOD_H = 24.0
START_HOUR, END_HOUR = 0.0, 240.0                     # day 0 -> day 10
HORIZON = 24                                          # prediction horizon, Sec. 4.1
CHANCE_LEVEL = 0.95                                   # alpha of the chance constraint, v2 (13f)/(14)
INV_NORMAL_CDF_95 = 1.6448536                         # Phi^-1(0.95)


def load_digitised(path=DATA_CSV):
    columns = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key and value not in (None, ''):
                    columns.setdefault(key, []).append(float(value))
    return {key: np.array(values) for key, values in columns.items()}


def water_consumption(hours, coefficients, exact=False):
    # demand model g_lambda/lambda_0 and g_mu of eq. (8), (11): the
    # instantaneous rate, and the interval consumption v_lambda / v_mu.
    # Default is the eq. (23c) sampled stand-in dt*g(t_k), the convention the
    # recorded paper-reproduction runs use; exact=True integrates the series
    # over [t_k, t_k + dt] as the exact antiderivative of eq. (18)
    hours = np.asarray(hours, dtype=float)
    w = 2 * np.pi / DEMAND_PERIOD_H
    rate = coefficients[0] * np.ones_like(hours)
    consumption = coefficients[0] * TIME_STEP * np.ones_like(hours)
    for n in range(1, (len(coefficients) - 1) // 2 + 1):
        a, b = coefficients[2 * n - 1], coefficients[2 * n]
        angle = n * w * hours
        rate += a * np.cos(angle) + b * np.sin(angle)
        if exact:
            consumption += (a * (np.sin(angle + n * w * TIME_STEP) - np.sin(angle))
                            - b * (np.cos(angle + n * w * TIME_STEP) - np.cos(angle))) / (n * w)
        else:
            consumption = TIME_STEP * rate
    return rate, consumption


def load_case(v_exact=False):
    with open(MODEL_JSON) as f:
        model = json.load(f)

    digitised = load_digitised()
    hour = digitised['hour']

    simulated = (hour >= START_HOUR) & (hour <= END_HOUR)
    hours = hour[simulated]
    if not np.any(hour == START_HOUR):
        raise ValueError('%s has no sample at hour %.0f' % (DATA_CSV, START_HOUR))
    if (END_HOUR - START_HOUR) % DEMAND_PERIOD_H != 0:
        raise ValueError('simulation window must be a whole number of days: '
                         'the closed-loop forecast is indexed modulo 24 h')

    demand1, consumption1 = water_consumption(hours, model['g_lambda'], exact=v_exact)
    demand2, consumption2 = water_consumption(hours, model['g_mu'], exact=v_exact)

    price_24h = [1.0] * 7 + [2.0] * 15 + [1.0] * 2    # price of Fig. 7, middle
    return {
        'friction': model['f_theta'],
        'n_harmonics': model['n_harmonics'],
        'hours': hours.copy(),
        'water_demand1_forecast': demand1,
        'water_demand2_forecast': demand2,
        'water_consumption1_forecast': consumption1,
        'water_consumption2_forecast': consumption2,
        'price': np.array([price_24h[int(h) % 24] for h in hours]),
        'water_level_min': WATER_LEVEL_MIN,
        'water_level_max': WATER_LEVEL_MAX,
        'water_level_start': float(digitised['level_m'][hour == START_HOUR][0]),
        'flow_max': float(np.max(digitised['flow_d1_m3h'])) + 20.0,
        'noise_std_zone1': float(np.std(digitised['water_demand1_m3h'][simulated] - demand1)),
        'noise_std_zone2': float(np.std(digitised['water_demand2_m3h'][simulated] - demand2)),
    }


def pressure_drop(theta, flow, water_demand1):
    # friction/head model f_theta of eq. (12) - represents pressure drop from the pump to the bottom of the tower
    abs_flow = ca.sqrt(flow * flow + 1e-5)
    deviation = ca.sqrt((flow - water_demand1) * (flow - water_demand1) + 1e-5)
    pressure_drop = (theta['th1'] * abs_flow * flow
                     + theta['th2'] * deviation * (flow - water_demand1)
                     + theta['th3'] * abs_flow * (flow - water_demand1)
                     + theta['th4'] * deviation * flow
                     + theta['th5'])
    return pressure_drop


def pump_power(theta, flow, water_demand1, water_level):
    # supply pressure of eq. (14c), (20)
    # pump power (p1 - p0)*d1*kW/eta of (14a)
    pressure_after_pump = pressure_drop(theta, flow, water_demand1) + LEVEL_PRESSURE_COEF * (water_level + TOWER_HEIGHT)
    pump_power = (pressure_after_pump - PRESSURE_BEFORE_PUMP) * flow * UNIT_CONVERSION / PUMP_EFFICIENCY
    return pressure_after_pump, pump_power


def make_mpc(theta, steps, flow_max, kappa_ramp=KAPPA_RAMP):
    # paper eq. (22): discrete-time form of (14) -- cost (14a), dynamics
    # (14b), pressure (14c), bounds (14d), over delta-t steps
    opti = ca.Opti()
    flow = opti.variable(steps)                 # d1 control action
    water_level = opti.variable(steps + 1)      # h state
    water_level_now = opti.parameter()          # h(t0): receding initialisation and terminal anchor of (14a)
    water_demand1 = opti.parameter(steps)       # d_bar fitted g_lambda/lambda_0 (8)/(10)
    water_demand2 = opti.parameter(steps)       # d_{n+1} fitted g_mu, (9)/(11)
    water_consumption1 = opti.parameter(steps)  # v_lambda of (18): water consumed in PZ1 over one interval, m3
    water_consumption2 = opti.parameter(steps)  # v_mu of (18)
    price = opti.parameter(steps)               # c(tau), (14a)
    level_margin = opti.parameter(steps)        # chance-constraint tightening, v2 (13f)/(14), set to 0 in open-loop

    opti.subject_to(water_level[0] == water_level_now)                            # receding horizon initialisation
    opti.subject_to(opti.bounded(WATER_LEVEL_MIN + level_margin, water_level[1:],
                                 WATER_LEVEL_MAX - level_margin))                   # (14d)/(23b)
    opti.subject_to(opti.bounded(0.0, flow, flow_max))                              # (14d)/(23b)

    cost = ca.MX(0)
    for k in range(steps):
        opti.subject_to(water_level[k + 1] == water_level[k]
                        + (TIME_STEP / AREA) * flow[k]
                        - (water_consumption1[k] + water_consumption2[k]) / AREA)       # (14b)/(19)
        _, power_k = pump_power(theta, flow[k], water_demand1[k], water_level[k + 1])   # (14c)/(20)*flow
        cost += price[k] * power_k                                                      # (14a)/(23a) running cost
    if kappa_ramp:
        cost += kappa_ramp * ca.sumsqr(flow[1:] - flow[:steps - 1])                     # (13a) ramp cost
    cost += KAPPA * (water_level[steps] - water_level_now) ** 2                          # (14a)/(23a) terminal cost
    exchange = 0.5 * ca.sum1(ca.fabs(TIME_STEP * flow - water_consumption1)
                             + water_consumption2) * DEMAND_PERIOD_H / (steps * TIME_STEP)
    opti.subject_to(exchange >= WATER_QUALITY)
    opti.minimize(cost)
    opti.solver('ipopt', {'ipopt.print_level': 0, 'ipopt.sb': 'yes',
                          'print_time': 0, 'show_eval_warnings': False})
    return {'opti': opti, 'flow': flow, 'water_level': water_level,
            'water_level_now': water_level_now,
            'water_demand1': water_demand1, 'water_demand2': water_demand2,
            'water_consumption1': water_consumption1, 'water_consumption2': water_consumption2,
            'price': price, 'level_margin': level_margin}


def solve_mpc(mpc, water_level_now, water_demand1, water_demand2, price,
              level_margin=None,
              water_consumption1=None, water_consumption2=None):
    # water_level_now anchors both the initial level and the terminal
    # periodicity target h(t0) of (14a); level_margin defaults to zero (v1
    # deterministic bounds)

    opti = mpc['opti']
    opti.set_value(mpc['water_level_now'], water_level_now)
    opti.set_value(mpc['water_demand1'], water_demand1)
    opti.set_value(mpc['water_demand2'], water_demand2)
    opti.set_value(mpc['price'], price)
    opti.set_value(mpc['water_consumption1'], water_consumption1)
    opti.set_value(mpc['water_consumption2'], water_consumption2)
    if level_margin is None:
        level_margin = np.zeros(len(np.atleast_1d(price)))
    opti.set_value(mpc['level_margin'], level_margin)
    solution = opti.solve()
    flows = np.asarray(solution.value(mpc['flow'])).ravel()
    water_levels = np.asarray(solution.value(mpc['water_level'])).ravel()
    return flows, water_levels


def step_env(water_level, flow, water_consumption1, water_consumption2):
    # env-side (18): the same model the controller plans with
    return water_level + (TIME_STEP / AREA) * flow - (water_consumption1 + water_consumption2) / AREA


def realize_demand(case, seed):
    # stochastic demand for the closed loop: independent white noise per zone,
    # std scaled with the mean demand at 2x the fit residual (v2 eq. (2)),
    # floored at zero; also returns the per-hour stds for the chance margins.
    # The realized consumption integrates the exact forecast profile of (18)
    # over the interval plus the same piecewise-constant noise as the rate
    std1 = 2.0 * case['noise_std_zone1'] * case['water_demand1_forecast'] \
        / np.mean(case['water_demand1_forecast'])
    std2 = 2.0 * case['noise_std_zone2'] * case['water_demand2_forecast'] \
        / np.mean(case['water_demand2_forecast'])
    rng = np.random.default_rng(seed)
    water_demand1 = np.maximum(case['water_demand1_forecast']
                               + rng.normal(0.0, 1.0, len(case['hours'])) * std1, 0.0)
    water_demand2 = np.maximum(case['water_demand2_forecast']
                               + rng.normal(0.0, 1.0, len(case['hours'])) * std2, 0.0)
    water_consumption1 = case['water_consumption1_forecast'] \
        + TIME_STEP * (water_demand1 - case['water_demand1_forecast'])
    water_consumption2 = case['water_consumption2_forecast'] \
        + TIME_STEP * (water_demand2 - case['water_demand2_forecast'])
    return water_demand1, water_demand2, water_consumption1, water_consumption2, std1, std2


def apply_safety(flow, water_level, water_consumption1, water_consumption2, flow_max):
    # local safety controller of v2 Sec. 3.2, simplified: hourly sampling and
    # the forecast consumption in place of the paper's Kalman-filter estimate
    demand_rate = (water_consumption1 + water_consumption2) / TIME_STEP
    flow_upper = demand_rate + (WATER_LEVEL_MAX - water_level) * AREA / TIME_STEP
    flow_lower = demand_rate + (WATER_LEVEL_MIN - water_level) * AREA / TIME_STEP
    safe = min(flow, flow_upper)              # overflow has priority
    safe = max(safe, min(flow_lower, flow_upper))
    safe = min(max(safe, 0.0), flow_max)
    return safe, abs(safe - flow) > 1e-9


def run_open_loop(case, theta, kappa_ramp=KAPPA_RAMP):
    # open loop: one MPC solve over the full window, nominal demand taken as
    # the truth, nothing replanned
    steps = len(case['hours'])
    mpc = make_mpc(theta, steps, case['flow_max'], kappa_ramp)
    flows, water_levels = solve_mpc(mpc, case['water_level_start'],
                                    case['water_demand1_forecast'], case['water_demand2_forecast'],
                                    case['price'],
                                    water_consumption1=case['water_consumption1_forecast'],
                                    water_consumption2=case['water_consumption2_forecast'])
    # p1 of eq. (2)/(20) and the power of (14a), evaluated at the solved
    # trajectory by the same expression the cost uses
    evaluated = [pump_power(theta, flows[k], case['water_demand1_forecast'][k],
                                water_levels[k + 1]) for k in range(steps)]
    pressures = np.array([float(value[0]) for value in evaluated])
    powers = np.array([float(value[1]) for value in evaluated])
    return water_levels[1:], flows, pressures, powers


def run_receding_horizon(case, theta, seed, chance_constraints=True, kappa_ramp=KAPPA_RAMP):
    # closed loop architecture: forecast -> solve_mpc (global controller,
    # chance-constrained unless disabled) -> safety projection (local
    # controller) -> step_env, the environment driven by the realized demand
    water_demand1_realized, water_demand2_realized, \
        water_consumption1_realized, water_consumption2_realized, std1, std2 = \
        realize_demand(case, seed)

    mpc = make_mpc(theta, HORIZON, case['flow_max'], kappa_ramp)
    water_levels, flows, pressures, powers = [], [], [], []
    interventions = 0
    water_level = case['water_level_start']
    for k in range(len(case['hours'])):
        window = np.arange(k, k + HORIZON) % 24
        # v2 (14), naive one-step-ahead variance of Remark 1: the level error
        # over one sample is (dt/area) * (demand error)
        level_margin = INV_NORMAL_CDF_95 * (TIME_STEP / AREA) \
            * np.sqrt(std1[window] ** 2 + std2[window] ** 2) \
            if chance_constraints else None
        planned_flows, planned_levels = solve_mpc(
            mpc, water_level, case['water_demand1_forecast'][window],
            case['water_demand2_forecast'][window], case['price'][window],
            level_margin=level_margin,
            water_consumption1=case['water_consumption1_forecast'][window],
            water_consumption2=case['water_consumption2_forecast'][window])
        flow = float(planned_flows[0])

        # warm start the next solve with the shifted plan
        mpc['opti'].set_initial(mpc['flow'],
                                np.append(planned_flows[1:], planned_flows[-1]))
        mpc['opti'].set_initial(mpc['water_level'],
                                np.append(planned_levels[1:], planned_levels[-1]))

        flow, intervened = apply_safety(flow, water_level,
                                        case['water_consumption1_forecast'][k],
                                        case['water_consumption2_forecast'][k],
                                        case['flow_max'])
        interventions += intervened

        water_level = step_env(water_level, flow,
                                 water_consumption1_realized[k],
                                 water_consumption2_realized[k])
        pressure, power_k = pump_power(theta, flow, water_demand1_realized[k], water_level)
        water_levels.append(water_level)
        flows.append(flow)
        pressures.append(float(pressure))
        powers.append(float(power_k))

    water_levels = np.array(water_levels)
    violations = int(np.sum((water_levels < WATER_LEVEL_MIN - 1e-6)
                            | (water_levels > WATER_LEVEL_MAX + 1e-6)))
    print('  seed %d: safety interventions %d h out of %d'
          % (seed, interventions, len(water_levels)))
    if violations:
        print('  warning: seed %d left the level bounds %d h out of %d '
              '(min %.3f, max %.3f)' % (seed, violations, len(water_levels),
                                        water_levels.min(), water_levels.max()))
    return water_demand1_realized, water_demand2_realized, water_levels, np.array(flows), \
        np.array(pressures), np.array(powers)


def write_csv(path, columns):
    names = list(columns)
    with open(path, 'w', newline='') as f:
        f.write(','.join(names) + '\n')
        for k in range(len(columns[names[0]])):
            f.write(','.join('%.4f' % columns[name][k] for name in names) + '\n')
    print('wrote %s (%d samples, %d columns)' % (path, len(columns[names[0]]),
                                                 len(names)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--receding', action='store_true',
                        help='24 h receding-horizon closed loop with '
                             'stochastic demand instead of the default '
                             'open-loop solve')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                        help='noise seeds for --receding (default: 1 2 3)')
    parser.add_argument('--no-chance', action='store_true',
                        help='disable the v2 chance-constraint tightening in '
                             'the receding-horizon MPC (v1 deterministic bounds)')
    parser.add_argument('--kappa-ramp', type=float, default=KAPPA_RAMP,
                        help='v2 (13a) control-ramp cost weight (default 0)')
    parser.add_argument('--v-exact', action='store_true',
                        help='eq. (18) exact interval integrals in the dynamics '
                             'instead of the default eq. (23c) stand-in dt*g(t_k)')
    parser.add_argument('--no-plot', action='store_true')
    arguments = parser.parse_args()

    case = load_case(v_exact=arguments.v_exact)
    print('model.json: N=%d harmonics, f_theta %s' % (
        case['n_harmonics'], {k: round(v, 6) for k, v in case['friction'].items()}))
    print('case: flow max %.1f m3/h, level start %.3f m, window %.0f..%.0f h, '
          'v_lambda %s, kappa_ramp %.3g'
          % (case['flow_max'], case['water_level_start'], START_HOUR, END_HOUR,
             'eq. 18 exact' if arguments.v_exact else 'eq. 23c stand-in',
             arguments.kappa_ramp))
    theta = case['friction']

    if arguments.receding:
        runs, columns = [], {'t_h': case['hours'], 'price': case['price'],
                             'water_demand1_forecast': case['water_demand1_forecast'],
                             'water_demand2_forecast': case['water_demand2_forecast']}
        for seed in arguments.seeds:
            water_demand1, water_demand2, water_levels, flows, pressures, powers = run_receding_horizon(
                case, theta, seed, chance_constraints=not arguments.no_chance,
                kappa_ramp=arguments.kappa_ramp)
            runs.append({'label': 'seed %d' % seed,
                         'water_demand1_realized': water_demand1,
                         'water_demand2_realized': water_demand2,
                         'water_levels': water_levels, 'flows': flows,
                         'pressures': pressures})
            for name, series in [('water_demand1_realized', water_demand1),
                                 ('water_demand2_realized', water_demand2),
                                 ('water_level', water_levels),
                                 ('pump_flow', flows), ('supply_pressure', pressures),
                                 ('pump_power', powers)]:
                columns['%s_s%d' % (name, seed)] = series
            print('seed %d: level [%.3f, %.3f] m, d1 [%.3f, %.3f] m3/h, '
                  'p1 [%.3f, %.3f] bar, power max %.3f mean %.3f kW'
                  % (seed, water_levels.min(), water_levels.max(),
                     flows.min(), flows.max(), pressures.min(), pressures.max(),
                     powers.max(), powers.mean()))
        suffix = '_receding'
        write_csv(os.path.join(FOLDER, 'kallesoe_trajectory%s.csv' % suffix), columns)
    else:
        water_levels, flows, pressures, powers = run_open_loop(case, theta, arguments.kappa_ramp)
        runs = [{'label': 'open loop',
                 'water_demand1_realized': case['water_demand1_forecast'],
                 'water_demand2_realized': case['water_demand2_forecast'],
                 'water_levels': water_levels,
                 'flows': flows, 'pressures': pressures}]
        suffix = '_openloop'
        write_csv(os.path.join(FOLDER, 'kallesoe_trajectory%s.csv' % suffix),
                  {'t_h': case['hours'], 'price': case['price'],
                   'water_demand1_forecast': case['water_demand1_forecast'],
                   'water_demand2_forecast': case['water_demand2_forecast'],
                   'water_level': water_levels, 'pump_flow': flows,
                   'supply_pressure_bar': pressures, 'pump_power_kw': powers})
        print('open loop: level [%.3f, %.3f] m, d1 [%.3f, %.3f] m3/h, '
              'p1 [%.3f, %.3f] bar, power max %.3f mean %.3f kW'
              % (water_levels.min(), water_levels.max(), flows.min(), flows.max(),
                 pressures.min(), pressures.max(), powers.max(), powers.mean()))

    if not arguments.no_plot:
        trajectory(os.path.join(FOLDER, 'kallesoe_trajectory%s.png' % suffix),
                   case, runs)


if __name__ == '__main__':
    main()