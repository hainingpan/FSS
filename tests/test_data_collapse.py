import numpy as np
import pandas as pd
import pytest
from fss import DataCollapse


def generate_pseudo_data(pc=0.5, p_list=np.round(np.linspace(0.45, 0.55, 11), 2),
                         nu=1, L_list=np.arange(10, 20, 2), beta=0.5,
                         f=lambda x: (1-x)**(1/2), seed=0, epsilon=0.01, N=100):
    """Generate pseudo data following y~L^{-beta/nu} f[(p-p_c)L^{1/nu}] + epsilon"""
    rng = np.random.default_rng(seed)
    data_dict = {(p, L): L**(-beta/nu)*f((p-pc)*L**(1/nu)) + rng.normal(0, epsilon, N)
                 for L in L_list for p in p_list}
    index = pd.MultiIndex.from_tuples(list(data_dict.keys()), names=['p', 'L'])
    df = pd.DataFrame({'observations': data_dict.values()}, index=index)
    return df


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
    rng = np.random.default_rng(42)
    p_c_true, nu_true, y_true = 0.5, 1.3, 1.0
    a_true = np.array([[1.0, 0.5], [0.3, 0.2]])  # (n1=1, n2=1)

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
    df = pd.DataFrame({'observations': list(data.values())}, index=index)

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
