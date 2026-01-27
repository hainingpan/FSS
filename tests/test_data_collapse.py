import matplotlib
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fss import DataCollapse
from fss.data_collapse import _append_subscript, _format_token


def generate_pseudo_data(pc=0.5, p_list=np.round(np.linspace(0.45, 0.55, 11), 2),
                         nu=1.0, L_list=np.arange(10, 20, 2), beta=0.5,
                         f=lambda x: (1-x)**(1/2), seed=0, epsilon=0.01, N=100):
    """Generate pseudo data following y~L^{-beta/nu} f[(p-p_c)L^{1/nu}] + epsilon"""
    rng = np.random.default_rng(seed)
    data_dict = {(p, L): L**(-beta/nu)*f((p-pc)*L**(1/nu)) + rng.normal(0, epsilon, N)
                 for L in L_list for p in p_list}
    index = pd.MultiIndex.from_tuples(list(data_dict.keys()), names=['p', 'L'])
    df = pd.DataFrame({'observations': data_dict.values()}, index=index)
    return df


def generate_drift_data(seed=42):
    rng = np.random.default_rng(seed)
    p_c_true, nu_true, y_true = 0.5, 1.3, 1.0
    a_true = np.array([[1.0, 0.5], [0.3, 0.2]])
    p_list = np.round(np.linspace(0.45, 0.55, 11), 2)
    L_list = np.array([8, 16, 32, 64])
    data = {}
    for L in L_list:
        for p in p_list:
            x = (p - p_c_true) * L ** (1 / nu_true)
            ir = L ** (-y_true)
            y_mean = sum(a_true[j1, j2] * x**j1 * ir**j2
                         for j1 in range(2) for j2 in range(2))
            data[(p, L)] = rng.normal(y_mean, 0.01, 100)
    index = pd.MultiIndex.from_tuples(list(data.keys()), names=['p', 'L'])
    return pd.DataFrame({'observations': list(data.values())}, index=index)


def test_datacollapse():
    """Test basic datacollapse recovers known parameters."""
    # Generate pseudo data with p_c=0.5, nu=1, beta=0.5 (from example.ipynb)
    p_c_true, nu_true, beta_true = 0.5, 1.0, 0.5
    df = generate_pseudo_data(pc=p_c_true, nu=nu_true, beta=beta_true)

    dc = DataCollapse(df, p_='p', L_='L', params={}, p_range=[0.45, 0.55])
    res = dc.datacollapse(p_c=0.505, nu=1.3, beta=0.0,
                          p_c_vary=True, nu_vary=True, beta_vary=True)

    assert abs(res.params['p_c'].value - p_c_true) < 0.05
    assert abs(res.params['nu'].value - nu_true) < 0.3
    assert abs(res.params['beta'].value - beta_true) < 0.3


def test_datacollapse_with_drift_GLS():
    """Test datacollapse_with_drift_GLS recovers known parameters."""
    p_c_true, nu_true, y_true = 0.5, 1.3, 1.0
    df = generate_drift_data()

    dc = DataCollapse(df, p_='p', L_='L', params={}, p_range=[0.45, 0.55])
    res = dc.datacollapse_with_drift_GLS(n1=1, n2=1, p_c=0.5, nu=1.0, y=1.0,
                                         p_c_range=(0.4, 0.6), nu_range=(0.5, 2.0))

    assert abs(res.params['p_c'].value - p_c_true) < 0.05
    assert abs(res.params['nu'].value - nu_true) < 0.3
    assert abs(res.params['y'].value - y_true) < 0.5


def test_parameter_sweep():
    """Test parameter_sweep finds correct parameters."""
    p_c_true, nu_true, beta_true = 0.5, 1.0, 0.0
    df = generate_pseudo_data(pc=p_c_true, nu=nu_true, beta=beta_true, epsilon=0.1)

    dc = DataCollapse(df, p_='p', L_='L', params={}, p_range=[0.45, 0.55])
    result = dc.parameter_sweep(
        p_c=np.linspace(0.48, 0.52, 10),
        nu=np.linspace(0.8, 1.2, 10),
        beta=0,
        n_jobs=1,
    )

    assert abs(result['p_c'] - p_c_true) < 0.05
    assert abs(result['nu'] - nu_true) < 0.3


def test_parameter_sweep_axes_limits_from_datalim():
    df = generate_pseudo_data(pc=0.5, nu=1.0, beta=0.0, epsilon=0.1)

    fig, ax = plt.subplots()
    dc = DataCollapse(df, p_='p', L_='L', params={}, p_range=[0.45, 0.55])
    p_c_values = np.linspace(0.48, 0.52, 4)
    nu_values = np.linspace(0.8, 1.2, 4)
    result = dc.parameter_sweep(
        p_c=p_c_values,
        nu=nu_values,
        beta=0,
        n_jobs=1,
        ax=ax,
    )

    def edges_from_centers(values):
        deltas = np.diff(values)
        first = values[0] - deltas[0] / 2
        last = values[-1] + deltas[-1] / 2
        mids = values[:-1] + deltas / 2
        return np.concatenate([[first], mids, [last]])

    def trim_nan_edges(values, grid, axis):
        if axis == 0:
            valid = ~np.all(np.isnan(grid), axis=1)
        else:
            valid = ~np.all(np.isnan(grid), axis=0)
        if not np.any(valid):
            return values, grid
        start = int(np.argmax(valid))
        end = int(len(valid) - np.argmax(valid[::-1]))
        if axis == 0:
            return values[start:end], grid[start:end, :]
        return values[start:end], grid[:, start:end]

    grid = result["chi2_grid"]
    trimmed_p, grid = trim_nan_edges(p_c_values, grid, axis=0)
    trimmed_nu, _ = trim_nan_edges(nu_values, grid, axis=1)
    x_edges = edges_from_centers(trimmed_nu)
    y_edges = edges_from_centers(trimmed_p)
    assert ax.get_xlim() == pytest.approx((x_edges[0], x_edges[-1]))
    assert ax.get_ylim() == pytest.approx((y_edges[0], y_edges[-1]))
    plt.close(fig)


def test_format_token_math_and_text():
    assert _format_token("p") == "$p$"
    assert _format_token("a_i") == "$a_i$"
    assert _format_token("eval_dropout_rate") == "eval_dropout_rate"
    assert _format_token("$\\nu$") == "$\\nu$"


def test_append_subscript_math_and_text():
    assert _append_subscript("a_i", "i") == "${a_i}_i$"
    assert _append_subscript("eval_dropout_rate", "i") == "eval_dropout_rate$_i$"
    assert _append_subscript("eval_dropout_rate", "c") == "eval_dropout_rate$_c$"
    assert _append_subscript("$\\nu$", "i") == "${\\nu}_i$"


def test_format_token_exponent_text():
    token = "eval_dropout_rate"
    formatted = _format_token(token)
    assert f"{formatted}$^{{\\beta/\\nu}}$" == "eval_dropout_rate$^{\\beta/\\nu}$"


def test_plot_data_collapse_drift_labels_math_L():
    df = generate_drift_data()
    dc = DataCollapse(df, p_='p', L_='L', params={}, p_range=[0.45, 0.55])
    dc.datacollapse_with_drift_GLS(
        n1=1,
        n2=1,
        p_c=0.5,
        nu=1.0,
        y=1.0,
        beta=0.5,
        p_c_range=(0.4, 0.6),
        nu_range=(0.5, 2.0),
    )
    fig, ax = plt.subplots()
    dc.plot_data_collapse(ax=ax, drift=True, driftcollapse=True, plot_irrelevant=True)
    ax2 = next(axis for axis in ax.figure.axes if axis is not ax)
    normalize = lambda value: " ".join(value.split())
    assert normalize(ax.get_ylabel()) == normalize(r"$(y_i - y_{irre}) L^{\beta/\nu}$")
    assert normalize(ax2.get_ylabel()) == normalize(r"$y_{irre} L^{\beta/\nu}$")
    plt.close(fig)


def test_plot_data_collapse_drift_labels_text_L():
    df = generate_drift_data()
    df.index = df.index.set_names(["p", "eval_dropout_rate"])
    dc = DataCollapse(df, p_='p', L_='eval_dropout_rate', params={}, p_range=[0.45, 0.55])
    dc.datacollapse_with_drift_GLS(
        n1=1,
        n2=1,
        p_c=0.5,
        nu=1.0,
        y=1.0,
        beta=0.5,
        p_c_range=(0.4, 0.6),
        nu_range=(0.5, 2.0),
    )
    fig, ax = plt.subplots()
    dc.plot_data_collapse(ax=ax, drift=True, driftcollapse=True, plot_irrelevant=True)
    ax2 = next(axis for axis in ax.figure.axes if axis is not ax)
    normalize = lambda value: " ".join(value.split())
    assert normalize(ax.get_ylabel()) == normalize(
        r"$(y_i - y_{irre})$ eval_dropout_rate$^{\beta/\nu}$"
    )
    assert normalize(ax2.get_ylabel()) == normalize(
        r"$y_{irre}$ eval_dropout_rate$^{\beta/\nu}$"
    )
    plt.close(fig)


def test_plot_data_collapse_title_pm_and_text_p():
    df = generate_pseudo_data(pc=0.5, nu=1.0, beta=0.5)
    df.index = df.index.set_names(["eval_dropout_rate", "L"])
    dc = DataCollapse(df, p_='eval_dropout_rate', L_='L', params={}, p_range=[0.45, 0.55])
    res = dc.datacollapse(p_c=0.505, nu=1.3, beta=0.0,
                          p_c_vary=True, nu_vary=True, beta_vary=True)
    res.params['p_c'].stderr = 0.01
    res.params['nu'].stderr = 0.02
    res.params['beta'].stderr = 0.03
    fig, ax = plt.subplots()
    dc.plot_data_collapse(ax=ax)
    title = ax.get_title()
    assert r"\pm" in title
    assert "eval_dropout_rate$_c$" in title
    plt.close(fig)

    res.params['beta'].stderr = None
    fig, ax = plt.subplots()
    dc.plot_data_collapse(ax=ax)
    title = ax.get_title()
    assert r"\pm" not in title
    plt.close(fig)
