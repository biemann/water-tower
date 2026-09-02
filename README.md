# Smart Water Software — Water-Tower Case Study (v1), official implementation

Reference implementation of the water-tower case study of

> C. S. Kallesoe, A. K. Nilsson, H. Madsen et al., *Smart Water Software: The
> Development of an Efficient Model Predictive Control System for Water
> Distribution Networks*, IFAC-PapersOnLine 50-1 (2017) 6582–6587.

The folder is self-contained: one script, the data extracted from the paper's
figures in `data/`, and every output written next to the script.  Equation
numbers below refer to the paper.

## Run

```sh
python fit_models.py                   # identify g_lambda, g_mu, f_theta -> data/model.json
python kallesoe_receding_horizon.py    # open-loop economic MPC + figure
python kallesoe_receding_horizon.py --receding        # 24 h receding-horizon closed loop,
                                                      # 3 stochastic realizations overlaid
python kallesoe_receding_horizon.py --no-plot         # skip the figure
```

Dependencies: `numpy`, `scipy`, `casadi` (with IPOPT), `matplotlib` (figure).

`fit_models.py` identifies both demand models and the friction model from the
digitised series and writes every fitted parameter to `data/model.json`,
which the run script loads.  A fitted `data/model.json` ships with the
folder, so fitting is only needed if you change the data or the fit.
The `f_theta` fit regresses `p1 - pn` with `pn = alpha*(h + h0)` evaluated
at the digitised level, per eq. (2), (12), (13); it reproduces the digitised
supply pressure to 0.07 bar RMSE.

## Model and its equations

The reduced network has one pump station (supply flow `d1`), one elevated
reservoir (level `h`, area `A = 400 m2`) and two pressure zones whose demands
are `dbar` (PZ1) and `dn1` (PZ2).  The code comments carry the equation
numbers; the map is:

| Paper | What | Where |
| --- | --- | --- |
| (8), (10) | PZ1 demand `dbar = g_lambda/lambda0`, truncated Fourier series, identified by least squares | `fit_models.py` -> `g_lambda` |
| (11) | PZ2 demand `dn1 = g_mu`, same Fourier structure | `fit_models.py` -> `g_mu` |
| (12), (13) | reduced network `p1 - pn = f_theta(d1, t)`, five-term friction/head, NNLS identification | `fit_models.py` -> `f_theta` |
| (2), (20) | `pn = alpha*(h + h0)`, `p1 = f_theta + alpha*h` | `pressure_p1`, MPC stage cost |
| (14a) | economic cost `int 2*c(t)*(p1 - p0)*d1/eta dt + kappa*(h(t0+T) - h(t0))^2` | `solve_mpc` objective |
| (14b), (18) | `h_k+1 = h_k + (dt/A)*(d1_k - dbar_k - dn1_k)` | `solve_mpc`, `run_receding_horizon` |
| (14c) | `p1 = f_theta(d1, t) + alpha*h` | `solve_mpc` |
| (14d) | `h_min <= h <= h_max`, `0 <= d1 <= D1_MAX` | solver bounds |
| (17) | water-exchange (quality) constraint | carried by the level band and the `kappa` terminal term, as in the paper's numerical section |

Fixed physics and identified constants: `alpha = 0.1`, `h0 = 30 m`,
`p0 = 2.408 bar`, `eta = 0.65`, `kappa = 0.0629`.  `D1_MAX` is derived from
the data as max(observed `d1`) + 20 m3/h (with the day-0 start this includes
the on/off startup spike, so the bound is generous; the MPC itself never
exceeds ~80 m3/h) and the initial level is the digitised level at hour 0.
The default Fourier order is 12 harmonics (`--harmonics N` to change); the
fit RMSE against the digitised demand is 1.15 m3/h (PZ1) and 0.34 (PZ2),
and the friction fit recovers th1 = 2.8e-5, th3 = 6.2e-5, th5 = 0.516 at
0.075 bar RMSE.

## The experiment (paper Sec. 4.1, Fig. 7)

The first two days of the paper's run identify the model; the controller then
takes over.  This script reproduces the controlled phase from the day-0
initial condition: at every hourly step the economic MPC solves (14) over a
24 h horizon from the measured level, applies the first control `d1(t0)`, and
the plant advances with the *realized* demand.  The realization is the
nominal forecast (8), (11) plus a correlated AR(1) perturbation, the
stochastic user-demand variation of the paper's Fig. 5.  The perturbation
standard deviation is the nominal-model residual on the digitised series
(about 1.4 m3/h for `dbar`, 0.4 for `dn1`) scaled by `NOISE_SCALE = 5`, i.e.
roughly the 15% variation visible in Fig. 5; the measured residual alone is
model mismatch and would be invisible.  The figure shows realized demands
(solid) against the nominal forecasts (dashed).  The price `c(t)` is the
paper's designed repeating
signal (Fig. 7, middle): high by day, low at night.

## Data

| File | Content |
| --- | --- |
| `data/kallesoe_hourly.csv` | Digitised hourly series of Figs. 6-7 of the paper: reservoir level, price, `dbar`, `dn1`, `d1`, supply pressure (241 h) |
| `data/model.json` | All fitted parameters: `g_lambda`, `g_mu` Fourier coefficients per eq. (8)/(11) and the `f_theta` weights per eq. (12)/(13), written by `--fit` |

## Output

`kallesoe_trajectory[_openloop|_seedN].csv` with columns
`t_h, price, dbar_nom, dn1_nom, dbar, dn1, level, d1, pressure_p1_bar,
power_kw`, plus a figure laid out like the paper's Fig. 7: reservoir level
with the min/max requirements on top, the price signal, the flows (`dbar`
blue, `dn1` green, supply `d1` red), and the supply pressure `p1` from (20)
as the fourth panel, matching the pressure identification of Fig. 6.
