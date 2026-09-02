"""Predictive control of the Kallesoe water-tower network.

Official implementation of the water-tower case study (v1) of

    C. S. Kallesoe, A. K. Nilsson, H. Madsen et al.,
    "Smart Water Software: The Development of an Efficient Model Predictive
    Control System for Water Distribution Networks", IFAC-PapersOnLine 50-1
    (2017) 6582-6587.

By default this runs the open-loop economic MPC over the full 10-day window
and plots the result.  Use --receding for the true closed-loop experiment
(24 h receding horizon, AR(1)-perturbed demand).  Equation numbers in the
comments refer to the paper.

    python kallesoe_receding_horizon.py               # open loop + figure
    python kallesoe_receding_horizon.py --receding    # closed loop, 3 seeds
    python kallesoe_receding_horizon.py --no-plot     # skip the figure

The model (g_lambda, g_mu, f_theta) is identified by fit_models.py and read
from data/model.json.  Requires: numpy, casadi (IPOPT), matplotlib (figure).
"""
import argparse
import csv
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, 'data')
OUT_DIR = _HERE

DATA_CSV = os.path.join(DATA, 'kallesoe_hourly.csv')
MODEL_JSON = os.path.join(DATA, 'model.json')

# ---- fixed physics and MPC constants
# H_MIN/H_MAX: level bounds hbar/hunderbar of eq. (14d)
# A = 400 m2: reservoir area (lambda0 = 1/A in eq. 9)
# ALPHA, H0_TWR: pressure-level scaling of eq. (2), pn = alpha*(h + h0)
# P0, KAPPA: pump base pressure and terminal-level weight of eq. (14a)
# KW, ETA: bar*m3/h -> kW conversion and pump efficiency of eq. (14a)
H_MIN, H_MAX = 1.0, 2.0
DT = 1.0
A = 400.0
ALPHA = 0.1
H0_TWR = 30.0
P0 = 2.408049
KAPPA = 0.062893
KW, ETA = 0.02778, 0.65
EPS = 1e-5

PERIOD_H = 24.0

# simulation window: hours 0..240 (day 0 -> day 10, the paper's full 10-day
# experiment).  The initial level is the digitised level at hour 0.
T0, T1 = 0.0, 240.0
D1MAX_FIT_END = 191.0
BAND_PAD = 20.0

# AR(1) lag coefficient phi = e^{-theta} at the 1 h step (fitted on Fig.5).
PHI_H = 0.0001394
# stochastic user-demand variation as in the paper's Fig. 5: the nominal
# demand plus a correlated statistical variation.  The residual of the
# nominal model on the digitised series (the measured std below, ~1.4 and
# ~0.4 m3/h) is mostly model mismatch and far too small to see, so the
# disturbance is scaled up (~7% of the PZ1 demand).  The scale keeps 3
# standard deviations inside the smallest nightly demands (dbar dips to
# ~14, dn1 to ~4.4 m3/h) so the realized demand stays positive; the clip
# below only guards the far tail.
NOISE_SCALE = 2.0
SEEDS = (1, 2, 3)   # realizations overlaid in the figure
SEED = SEEDS[0]
HORIZON = 24

# designed price c(t) of the paper's Fig. 7 (middle plot), read off the
# digitised series: low at night (h 0..6 and 22..23), high by day (h 7..21)
PRICE_24 = [1.0] * 7 + [2.0] * 15 + [1.0] * 2

M = None   # scenario, set by main() before any run/plot call


# ----------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------
def load_digitised(path=DATA_CSV):
    cols = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key and val not in (None, ''):
                    cols.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in cols.items()}


def eval_fourier(t_h, coeffs):
    # evaluate g_lambda / g_mu (eq. 8) with the coefficients from model.json
    t_h = np.asarray(t_h, dtype=float)
    y = coeffs[0] * np.ones_like(t_h)
    for n in range(1, (len(coeffs) - 1) // 2 + 1):
        y += coeffs[2 * n - 1] * np.cos(2 * np.pi * n * t_h / PERIOD_H)
        y += coeffs[2 * n] * np.sin(2 * np.pi * n * t_h / PERIOD_H)
    return y


# ----------------------------------------------------------------------
# scenario: everything the MPC run needs, derived once from the data
# ----------------------------------------------------------------------
def load_scenario():
    global M
    with open(MODEL_JSON) as f:
        model = json.load(f)
    gl = np.array(model['g_lambda'])
    gm = np.array(model['g_mu'])

    fig = load_digitised()
    hour = fig['hour']

    # pump capacity from the observed supply flow, level start from the data
    win = (hour >= T0) & (hour <= D1MAX_FIT_END)
    d1_max = float(np.max(fig['flow_d1_m3h'][win]) + BAND_PAD)
    h_init = float(fig['level_m'][hour == T0][0])

    sim = (hour >= T0) & (hour <= T1)
    t = hour[sim]
    dbar_nom, dn1_nom = eval_fourier(t, gl), eval_fourier(t, gm)
    dist_std_dbar = float(np.std(fig['flow_dbar_m3h'][sim] - dbar_nom))
    dist_std_dn1 = float(np.std(fig['flow_dn1_m3h'][sim] - dn1_nom))

    n = int(T1 - T0) + 1
    t_h = T0 + np.arange(n, dtype=float)
    M = type('Scenario', (), {})()
    M.th = model['f_theta']
    M.t_h = t_h
    M.dbar_nom = eval_fourier(t_h, gl)
    M.dn1_nom = eval_fourier(t_h, gm)
    M.price = np.array([PRICE_24[int(h) % 24] for h in t_h])
    M.d1_max, M.h_init = d1_max, h_init
    M.dist_std_dbar, M.dist_std_dn1 = dist_std_dbar, dist_std_dn1
    print('model.json: g_lambda/g_mu with N=%d harmonics, f_theta %s'
          % (model['n_harmonics'],
             {k: round(v, 6) for k, v in M.th.items()}))
    print('scenario: D1_MAX = %.4f m3/h, H_INIT = %.4f m, window %.0f..%.0f h'
          % (d1_max, h_init, T0, T1))
    print('demand disturbance std (digitised - nominal): dbar %.3f, dn1 %.3f m3/h'
          % (dist_std_dbar, dist_std_dn1))


# ----------------------------------------------------------------------
# model evaluation (eq. 12, 2, 20)
# ----------------------------------------------------------------------
def ftheta_numeric(u, db):
    sus = np.sqrt(u * u + EPS)
    diff = np.sqrt((u - db) ** 2 + EPS)
    return (M.th['th1'] * sus * u + M.th['th2'] * diff * (u - db)
            + M.th['th3'] * sus * (u - db) + M.th['th4'] * diff * u
            + M.th['th5'])


def pressure_p1(u, db, h_next):
    """Supply pressure p1 from (20) with the reservoir pressure (2)."""
    return ftheta_numeric(u, db) + ALPHA * h_next + ALPHA * H0_TWR


def power_kw(u, db, h_next):
    return (pressure_p1(u, db, h_next) - P0) * u * KW / ETA


def ar1_noise(n, phi_h, std, seed):
    rng = np.random.default_rng(seed)
    eps_sd = std * np.sqrt(1.0 - phi_h ** 2)
    z = np.empty(n)
    z[0] = rng.normal(0.0, std)
    for i in range(1, n):
        z[i] = phi_h * z[i - 1] + rng.normal(0.0, eps_sd)
    return z


# ----------------------------------------------------------------------
# economic MPC (eq. 14) and the two experiment modes
# ----------------------------------------------------------------------
def solve_mpc(h_start, dbar_h, dn1_h, pr_h, h_ref):
    """One economic-MPC solve over the given horizon slice; returns all u, h."""
    import casadi as ca
    horizon = len(dbar_h)
    w, g = [], []
    lbw, ubw, w0, lbg, ubg = [], [], [], [], []
    J = ca.MX(0)
    h = h_start
    for k in range(horizon):
        db, dnn, pr = float(dbar_h[k]), float(dn1_h[k]), float(pr_h[k])
        u = ca.MX.sym('u%d' % k)
        w.append(u)
        lbw.append(0.0)
        ubw.append(M.d1_max)
        w0.append(min(max(db + dnn, 0.0), M.d1_max))

        sus = ca.sqrt(u * u + EPS)
        diff = ca.sqrt((u - db) * (u - db) + EPS)
        fth = (M.th['th1'] * sus * u + M.th['th2'] * diff * (u - db)
               + M.th['th3'] * sus * (u - db) + M.th['th4'] * diff * u
               + M.th['th5'])
        hn = h + (DT / A) * (u - db - dnn)                       # eq. (18)
        p1 = fth + ALPHA * hn + ALPHA * H0_TWR                   # eq. (20)
        J += 2.0 * pr * (p1 - P0) * u + 1e-6 * u * u             # eq. (14a)

        hnext = ca.MX.sym('h%d' % (k + 1))
        w.append(hnext)
        lbw.append(H_MIN)
        ubw.append(H_MAX)
        w0.append(min(max(h_start, H_MIN + 1e-3), H_MAX - 1e-3))
        g.append(h + (DT / A) * (u - db - dnn) - hnext)          # eq. (14b)
        lbg.append(0.0)
        ubg.append(0.0)
        h = hnext
    J += KAPPA * (h - h_ref) * (h - h_ref)

    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
            'show_eval_warnings': False}
    solver = ca.nlpsol('solver', 'ipopt',
                       {'x': ca.vertcat(*w), 'f': J, 'g': ca.vertcat(*g)}, opts)
    res = solver(x0=ca.DM(w0), lbx=ca.DM(lbw), ubx=ca.DM(ubw),
                 lbg=ca.DM(lbg), ubg=ca.DM(ubg))
    xv = np.asarray(res['x']).ravel()
    return xv[0::2], np.concatenate([[float(h_start)], xv[1::2]])


def run_receding_horizon(seed):
    """Closed loop: forecast nominal demand, realize AR(1)-perturbed demand."""
    z = ar1_noise(len(M.t_h), PHI_H, NOISE_SCALE * M.dist_std_dbar, seed)
    dbar_real = np.maximum(M.dbar_nom + z, 0.0)
    dn1_real = np.maximum(
        M.dn1_nom + NOISE_SCALE * M.dist_std_dn1 / M.dist_std_dbar * z, 0.0)

    level, d1, p1s, pwr = [], [], [], []
    h = M.h_init
    for k in range(len(M.t_h)):
        j = min(k + HORIZON, len(M.t_h))
        us, _ = solve_mpc(h, M.dbar_nom[k:j], M.dn1_nom[k:j], M.price[k:j],
                          M.h_init)
        u0 = float(us[0])

        h_next = min(max(h + (DT / A) * (u0 - dbar_real[k] - dn1_real[k]),
                         H_MIN), H_MAX)
        level.append(h_next)
        d1.append(u0)
        p1s.append(pressure_p1(u0, dbar_real[k], h_next))
        pwr.append(power_kw(u0, dbar_real[k], h_next))
        h = h_next
    return dbar_real, dn1_real, np.array(level), np.array(d1), \
        np.array(p1s), np.array(pwr)


def run_gate():
    """Deterministic open-loop: one full-window NLP (sanity check of the model)."""
    import casadi as ca
    n = len(M.t_h)
    w, g = [], []
    lbw, ubw, w0, lbg, ubg = [], [], [], [], []
    J = ca.MX(0)
    h = ca.MX.sym('h0')
    w.append(h)
    lbw.append(M.h_init)
    ubw.append(M.h_init)
    w0.append(M.h_init)
    for k in range(n):
        db, dnn, pr = M.dbar_nom[k], M.dn1_nom[k], M.price[k]
        u = ca.MX.sym('u%d' % k)
        w.append(u)
        lbw.append(0.0)
        ubw.append(M.d1_max)
        w0.append(min(max(db + dnn, 0.0), M.d1_max))
        sus = ca.sqrt(u * u + EPS)
        diff = ca.sqrt((u - db) * (u - db) + EPS)
        fth = (M.th['th1'] * sus * u + M.th['th2'] * diff * (u - db)
               + M.th['th3'] * sus * (u - db) + M.th['th4'] * diff * u
               + M.th['th5'])
        hn = h + (DT / A) * (u - db - dnn)
        p1 = fth + ALPHA * hn + ALPHA * H0_TWR
        J += 2.0 * pr * (p1 - P0) * u + 1e-6 * u * u
        hnext = ca.MX.sym('h%d' % (k + 1))
        w.append(hnext)
        lbw.append(H_MIN)
        ubw.append(H_MAX)
        w0.append(min(max(M.h_init, H_MIN + 1e-3), H_MAX - 1e-3))
        g.append(h + (DT / A) * (u - db - dnn) - hnext)
        lbg.append(0.0)
        ubg.append(0.0)
        h = hnext
    J += KAPPA * (h - M.h_init) * (h - M.h_init)

    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0,
            'show_eval_warnings': False}
    solver = ca.nlpsol('solver', 'ipopt',
                       {'x': ca.vertcat(*w), 'f': J, 'g': ca.vertcat(*g)}, opts)
    res = solver(x0=ca.DM(w0), lbx=ca.DM(lbw), ubx=ca.DM(ubw),
                 lbg=ca.DM(lbg), ubg=ca.DM(ubg))
    xv = np.asarray(res['x']).ravel()
    hs = np.concatenate([[M.h_init], xv[2::2]])
    us = xv[1::2]
    p1s = np.array([pressure_p1(us[k], M.dbar_nom[k], hs[k + 1])
                    for k in range(n)])
    pwr = np.array([power_kw(us[k], M.dbar_nom[k], hs[k + 1])
                    for k in range(n)])
    return hs[1:], us, p1s, pwr


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def write_csv(path, dbar, dn1, level, d1, p1s, pwr):
    with open(path, 'w', newline='') as f:
        f.write('t_h,price,dbar_nom,dn1_nom,dbar,dn1,level,d1,pressure_p1_bar,power_kw\n')
        for i in range(len(M.t_h)):
            f.write('%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.3f,%.4f,%.3f\n'
                    % (M.t_h[i], M.price[i], M.dbar_nom[i], M.dn1_nom[i],
                       dbar[i], dn1[i], level[i], d1[i], p1s[i], pwr[i]))
    print('wrote %s (%d samples)' % (path, len(M.t_h)))


def plot(path, runs):
    """Fig. 7-style result figure (level with the min/max level requirements,
    price, flows) extended with the supply pressure p1 from (20) as the
    fourth axis, as identified in the paper's Fig. 6 (bottom).  `runs` is a
    list of (label, dbar, dn1, level, d1, p1s) tuples; several stochastic
    realizations are overlaid."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping the plot')
        return
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    day = M.t_h / 24.0
    # fixed y-limits per panel, identical for every figure this script writes
    ylims = [(0.0, 2.5), (0.5, 2.5), (0, 100), (2.0, 5.0)]

    ax = axes[0]
    ax.axhline(H_MIN, color='red', lw=1.0, ls='--')
    ax.axhline(H_MAX, color='red', lw=1.0, ls='--')
    ax.axhline(M.h_init, color='gray', lw=0.8, ls=':')
    for r, (label, dbar, dn1, level, d1, p1s) in enumerate(runs):
        ax.plot(day, level, color='black', lw=1.0, alpha=0.45,
                label='h(t)' if r == 0 else None)
    ax.set_ylabel('level [m]')
    ax.set_ylim(*ylims[0])
    ax.text(day[0] + 0.1, H_MIN + 0.02, '$h_{min}$', color='red')
    ax.text(day[0] + 0.1, H_MAX - 0.09, '$h_{max}$', color='red')

    ax = axes[1]
    ax.plot(day, M.price, drawstyle='steps-mid', color='black', lw=1.2)
    ax.set_ylabel('price [-]')
    ax.set_ylim(*ylims[1])

    ax = axes[2]
    ax.plot(day, M.dbar_nom, color='blue', lw=1.0, ls=':', alpha=1.0,
            label=r'$\bar{d}=g_\lambda/\lambda_0$ nominal')
    ax.plot(day, M.dn1_nom, color='green', lw=1.0, ls=':', alpha=1.0,
            label=r'$d_{n+1}=g_\mu$ nominal')
    for r, (label, dbar, dn1, level, d1, p1s) in enumerate(runs):
        ax.plot(day, dbar, color='blue', lw=0.9, alpha=0.45,
                label='PZ1 realized' if r == 0 else None)
        ax.plot(day, dn1, color='green', lw=0.9, alpha=0.45,
                label='PZ2 realized' if r == 0 else None)
        ax.plot(day, d1, color='red', lw=0.9, alpha=0.45,
                label=r'$d_1$ (supply, MPC)' if r == 0 else None)
    ax.set_ylabel('flow [m3/h]')
    ax.set_ylim(*ylims[2])
    ax.legend(fontsize=7, ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.02))

    ax = axes[3]
    for r, (label, dbar, dn1, level, d1, p1s) in enumerate(runs):
        ax.plot(day, p1s, color='black', lw=1.0, alpha=0.45,
                label=label if r == 0 else None)
    ax.set_ylabel(r'$p_1$ [bar]')
    ax.set_ylim(*ylims[3])
    ax.set_xlabel('time [days]')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(day[0], day[-1])
    fig.suptitle('Predictive control of the Kallesoe network '
                 '(paper Fig. 7 layout)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    print('wrote %s' % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--receding', action='store_true',
                    help='24 h receding-horizon closed loop with '
                         'AR(1)-perturbed demand instead of the default '
                         'open-loop solve')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    load_scenario()

    if args.receding:
        runs, suffix = [], '_seeds%d%d%d' % SEEDS
        for seed in SEEDS:
            dbar, dn1, level, d1, p1s, pwr = run_receding_horizon(seed)
            runs.append(('seed %d' % seed, dbar, dn1, level, d1, p1s))
            write_csv(os.path.join(OUT_DIR, 'kallesoe_trajectory_seed%d.csv' % seed),
                      dbar, dn1, level, d1, p1s, pwr)
            print('seed %d: level [%.3f, %.3f] m, d1 [%.3f, %.3f] m3/h, '
                  'p1 [%.3f, %.3f] bar, power max %.3f mean %.3f kW'
                  % (seed, level.min(), level.max(), d1.min(), d1.max(),
                     p1s.min(), p1s.max(), pwr.max(), pwr.mean()))
        tag = 'receding horizon, seeds %s' % (SEEDS,)
    else:
        level, d1, p1s, pwr = run_gate()
        dbar, dn1 = M.dbar_nom, M.dn1_nom
        runs = [('open loop', dbar, dn1, level, d1, p1s)]
        suffix, tag = '_openloop', 'open-loop'
        write_csv(os.path.join(OUT_DIR, 'kallesoe_trajectory%s.csv' % suffix),
                  dbar, dn1, level, d1, p1s, pwr)

    if not args.no_plot:
        plot(os.path.join(OUT_DIR, 'kallesoe_trajectory%s.png' % suffix), runs)


if __name__ == '__main__':
    main()


