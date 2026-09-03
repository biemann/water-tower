"""Figure output for kallesoe_mpc.py."""


def trajectory(path, case, runs):
    # figure in the layout of the paper's Fig. 7, plus the supply pressure
    # p1 identified in the paper's Fig. 6 (bottom)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available; skipping the plot')
        return
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    days = case['hours'] / 24.0
    axis_limits = [(0.0, 2.5), (0.5, 2.5), (0, 100), (2.0, 5.0)]

    ax = axes[0]
    ax.axhline(case['water_level_min'], color='red', lw=1.0, ls='--')
    ax.axhline(case['water_level_max'], color='red', lw=1.0, ls='--')
    ax.axhline(case['water_level_start'], color='gray', lw=0.8, ls=':')
    for r, run in enumerate(runs):
        ax.plot(days, run['water_levels'], color='black', lw=1.0, alpha=0.45,
                label='h(t)' if r == 0 else None)
    ax.set_ylabel('level [m]')
    ax.set_ylim(*axis_limits[0])
    ax.text(days[0] + 0.1, case['water_level_min'] + 0.02, '$h_{min}$', color='red')
    ax.text(days[0] + 0.1, case['water_level_max'] - 0.09, '$h_{max}$', color='red')

    ax = axes[1]
    ax.plot(days, case['price'], drawstyle='steps-mid', color='black', lw=1.2)
    ax.set_ylabel('price [-]')
    ax.set_ylim(*axis_limits[1])

    ax = axes[2]
    ax.plot(days, case['water_demand1_forecast'], color='blue', lw=1.0, ls=':', alpha=1.0,
            label='water_demand1 forecast')
    ax.plot(days, case['water_demand2_forecast'], color='green', lw=1.0, ls=':', alpha=1.0,
            label='water_demand2 forecast')
    for r, run in enumerate(runs):
        ax.plot(days, run['water_demand1_realized'], color='blue', lw=0.9, alpha=0.45,
                label='water_demand1 realized' if r == 0 else None)
        ax.plot(days, run['water_demand2_realized'], color='green', lw=0.9, alpha=0.45,
                label='water_demand2 realized' if r == 0 else None)
        ax.plot(days, run['flows'], color='red', lw=0.9, alpha=0.45,
                label='pump_flow (MPC)' if r == 0 else None)
    ax.set_ylabel('flow [m3/h]')
    ax.set_ylim(*axis_limits[2])
    ax.legend(fontsize=7, ncol=3, loc='lower left', bbox_to_anchor=(0.0, 1.02))

    ax = axes[3]
    for r, run in enumerate(runs):
        ax.plot(days, run['pressures'], color='black', lw=1.0, alpha=0.45,
                label=run['label'] if r == 0 else None)
    ax.set_ylabel(r'$p_1$ [bar]')
    ax.set_ylim(*axis_limits[3])
    ax.set_xlabel('time [days]')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(days[0], days[-1])
    figure.suptitle('Predictive control of the Kallesoe network '
                    '(paper Fig. 7 layout)')
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=150)
    print('wrote %s' % path)
