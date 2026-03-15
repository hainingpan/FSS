"""Core finite-size scaling routines for data collapse analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, Any
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from lmfit.minimizer import MinimizerResult

__all__ = [
    "DataCollapse",
    "grid_search",
    "plot_chi2_ratio",
    "extrapolate_fitting",
    "plot_extrapolate_fitting",
    "optimal_df",
    "bootstrapping",
]

_MATH_TOKEN_RE = re.compile(r"^(?:[A-Za-z]|\\[A-Za-z]+)(?:_(?:[A-Za-z]|\\[A-Za-z]+))?$")


def _strip_math_delimiters(token: str) -> tuple[str, bool]:
    if token.startswith("$") and token.endswith("$") and len(token) >= 2:
        return token[1:-1], True
    return token, False


def _is_simple_math_token(token: str) -> bool:
    return _MATH_TOKEN_RE.match(token) is not None


def _format_token(token: str) -> str:
    token, forced_math = _strip_math_delimiters(token)
    if forced_math or _is_simple_math_token(token):
        return f"${token}$"
    return token


def _append_subscript(token: str, subscript: str) -> str:
    token, forced_math = _strip_math_delimiters(token)
    if forced_math or _is_simple_math_token(token):
        return f"${{{token}}}_{subscript}$"
    return f"{token}$_{subscript}$"


def _format_exponent(token: str, exponent: str) -> tuple[str, bool]:
    formatted = _format_token(token)
    formatted, forced_math = _strip_math_delimiters(formatted)
    if forced_math:
        return f"{formatted}^{{{exponent}}}", True
    return f"{formatted}$^{{{exponent}}}$", False


def _math_safe(token: str) -> str:
    """Make a user token safe for embedding inside a $...$ math expression.

    Simple math tokens (single letter, LaTeX commands, single subscript) are
    returned unchanged.  Anything else (e.g. ``train_size``) is wrapped in
    ``\\mathrm{...}`` with underscores replaced by ``\\_`` so that LaTeX
    does not interpret them as subscript operators.
    """
    raw, _ = _strip_math_delimiters(token)
    if _is_simple_math_token(raw):
        return raw
    return r'\mathrm{' + raw.replace('_', r'\_') + '}'

class DataCollapse:
    """Finite-size scaling data collapse analysis using lmfit optimization."""

    def __init__(
        self,
        df: pd.DataFrame,
        p_: str,
        L_: str,
        params: dict[str, Any] | None = None,
        p_range: list[float] = [-0.1, 0.1],
        Lmin: int | None = None,
        Lmax: int | None = None,
        adaptive_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        estimator: str = 'mean',
    ) -> None:
        """Perform finite-size scaling following y ~ L^{-beta/nu} f[(p-p_c)L^{1/nu}].

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MultiIndex containing at least `p_` and `L_` as level names.
            Must have an 'observations' column (for 'mean'/'std' estimator) or
            'estimator' and 'standard_error' columns (for 'manual' estimator).
        p_ : str
            Name of the tuning parameter index level (e.g., 'p', 'T', 'B').
        L_ : str
            Name of the system size index level (e.g., 'L', 't').
        params : dict[str, Any] | None, optional
            Fixed parameter values to select from the DataFrame via `xs()`.
            Example: `{'Metrics': 'O'}` to select rows where Metrics='O'.
        p_range : list[float], optional
            Range [min, max] of tuning parameter to include, by default [-0.1, 0.1].
        Lmin : int | None, optional
            Minimum system size to include. If None, no lower bound.
        Lmax : int | None, optional
            Maximum system size to include. If None, no upper bound.
        adaptive_func : Callable[[ndarray, ndarray], ndarray] | None, optional
            Custom function `f(p_values, L_values) -> values` for data selection.
            When provided, `p_range` filters on `f(p, L)` instead of `p` directly.
        estimator : str, optional
            Method for computing the estimator and standard error:
            - 'mean': Use mean of observations; standard error = std / sqrt(n).
            - 'std': Use std of observations; standard error = 1.
            - 'manual': Use 'estimator' and 'standard_error' columns directly.
        """
        self.p_range=p_range
        self.Lmin=0 if Lmin is None else Lmin
        self.Lmax=np.inf if Lmax is None else Lmax
        self.params=params
        self.p_=p_
        self.L_=L_
        self.adaptive_func=adaptive_func
        self.estimator=estimator
        self.df=self.load_dataframe(df,params)
        self.L_i,self.p_i,self.d_i,self.y_i = self.load_data()
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        params: dict[str, Any] | None,
    ) -> pd.DataFrame:
        """Filter and prepare the input DataFrame for analysis.

        Applies parameter selection via `xs()`, filters by system size bounds
        (Lmin, Lmax), and filters by tuning parameter range (p_range).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with MultiIndex.
        params : dict[str, Any] | None
            Parameter values to select via `df.xs()`.

        Returns
        -------
        pd.DataFrame
            Filtered and sorted DataFrame.
        """
        if params is None or len(params)==0:
            df=df
        else:
            df=df.xs(params.values(),level=list(params.keys()))
        df=df[(df.index.get_level_values(self.L_)<=self.Lmax) & (self.Lmin<=df.index.get_level_values(self.L_))]
        if self.adaptive_func is None:
            df=df[(df.index.get_level_values(self.p_)<=self.p_range[1]) & (self.p_range[0]<=df.index.get_level_values(self.p_))]
        else:
            val=self.adaptive_func(df.index.get_level_values(self.p_),df.index.get_level_values(self.L_))
            df=df[(val <=self.p_range[1]) & (self.p_range[0]<=val)]
        return df.sort_index(level=[self.L_,self.p_])

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract arrays from the filtered DataFrame.

        Computes the estimator (y_i) and standard error (d_i) based on the
        `estimator` setting ('mean', 'std', or 'manual').

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (L_i, p_i, d_i, y_i) where:
            - L_i: System sizes
            - p_i: Tuning parameter values
            - d_i: Standard errors
            - y_i: Estimator values (observable)

        Raises
        ------
        NotImplementedError
            If `estimator` is not one of 'mean', 'std', 'manual'.
        AssertionError
            If fewer than 4 unique tuning parameter values are present.
        """
        L_i=(self.df.index.get_level_values(self.L_).values)
        p_i=(self.df.index.get_level_values(self.p_).values)
        if self.estimator=='mean':
            d_i=(self.df['observations'].apply(np.std).values)/np.sqrt(self.df['observations'].apply(len).values)
            y_i=(self.df['observations'].apply(np.mean).values)
        elif self.estimator=='std':
            # d_i=(self.df['observations'].apply(lambda x: np.std(x)*np.sqrt(2/(len(x)))).values)
            d_i=(self.df['observations'].apply(lambda x: 1).values)
            y_i=(self.df['observations'].apply(np.std).values)
        elif self.estimator=='manual':
            d_i=(self.df['standard_error'].values)
            y_i=(self.df['estimator'].values)
        else:
            raise NotImplementedError(f'estimator {self.estimator} not implemented')
        if np.unique(p_i).shape[0]<4:
            warnings.warn(f'not enough data points {np.unique(p_i).shape[0]}', stacklevel=2)
        return L_i,p_i,d_i,y_i   

    
    def _smoothness_residuals(self, x_i, y_scaled, d_scaled):
        ind=np.argsort(x_i)
        x_i=x_i[ind]
        y_scaled=y_scaled[ind]
        d_scaled=d_scaled[ind]
        x={i:x_i[1+i:x_i.shape[0]-1+i] for i in [-1,0,1]}
        d={i:d_scaled[1+i:d_scaled.shape[0]-1+i] for i in [-1,0,1]}
        y={i:y_scaled[1+i:y_scaled.shape[0]-1+i] for i in [-1,0,1]}
        x_post_ratio=(x[1]-x[0])/(x[1]-x[-1])
        x_pre_ratio=(x[-1]-x[0])/(x[1]-x[-1])
        y_var=d[0]**2+(x_post_ratio*d[-1])**2+(x_pre_ratio*d[1])**2
        y_var=np.clip(y_var,y_var[y_var>0].min(),None)
        y_bar=x_post_ratio*y[-1]-x_pre_ratio*y[1]
        return (y[0]-y_bar)/np.sqrt(y_var)

    def _gls_solve(self, x_i, y_scaled, d_scaled, ir_i, n1, n2):
        j1,j2=np.meshgrid(np.arange(n1+1),np.arange(n2+1),indexing='ij')
        self.X=(x_i**j1.flatten()[:,np.newaxis] * ir_i**j2.flatten()[:,np.newaxis]).T
        Sigma_inv=np.diag(1/d_scaled**2)
        XX=self.X.T@ Sigma_inv @ self.X
        XY=self.X.T@ Sigma_inv @ y_scaled
        self.coeffs=np.linalg.inv(XX)@XY
        self.y_i_scaled=y_scaled
        self.d_i_scaled=d_scaled
        self.y_i_fitted=self.X @ self.coeffs
        return (self.y_i_scaled-self.y_i_fitted)/self.d_i_scaled

    def loss(self, p_c: float, nu: float, beta: float = 0) -> np.ndarray:
        """Compute smoothness-based residuals for data collapse.

        Evaluates how well data collapses onto a universal curve using linear
        interpolation. The scaling form is y ~ L^{-beta/nu} f((p-p_c)L^{1/nu}).

        Parameters
        ----------
        p_c : float
            Critical point of the tuning parameter.
        nu : float
            Correlation length exponent.
        beta : float, optional
            Order parameter exponent, by default 0.

        Returns
        -------
        np.ndarray
            Normalized residuals measuring deviation from smooth collapse.
        """
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        y_scaled=self.y_i * self.L_i**(beta/nu)
        d_scaled=self.d_i * self.L_i**(beta/nu)
        return self._smoothness_residuals(x_i, y_scaled, d_scaled)

    def loss_with_drift(
        self,
        p_c: float,
        nu: float,
        y: float,
        b1: list[float] | np.ndarray,
        b2: list[float] | np.ndarray,
        a: np.ndarray,
    ) -> np.ndarray:
        """Compute residuals with irrelevant scaling corrections (Taylor expansion).

        Uses a polynomial expansion for both relevant and irrelevant scaling
        variables. The scaling form includes corrections to scaling.

        Convention from  Sec. 2.5 from 10.1088/1367-2630/16/1/015012

        Parameters
        ----------
        p_c : float
            Critical point of the tuning parameter.
        nu : float
            Relevant critical exponent (correlation length exponent).
        y : float
            Irrelevant scaling exponent.
        b1 : list[float] | np.ndarray
            Coefficients for relevant scaling variable, shape (m1+1,).
            b1 = [b10=0, b11, b12, ...] where b10=0 is enforced to ensure u1(w=0)=0.
            Computes: u1 = sum_i b1[i] * w^i, where w = p - p_c.
        b2 : list[float] | np.ndarray
            Coefficients for irrelevant scaling variable, shape (m2+1,).
            b2 = [b20, b21, b22, ...].
            Computes: u2 = sum_i b2[i] * w^i.
        a : np.ndarray
            Coefficient matrix (n1+1, n2+1) for the Taylor expansion::

                a = [[a00,   a01=1, a02, a03, ...],
                     [a10=1, a11,   a12, a13, ...],
                     [a20,   a21,   a22, a23, ...],
                     ...]

            Note: a[0,1] = 1 and a[1,0] = 1 are fixed normalization constraints.
            Computes: f = sum_{i,j} a[i,j] * phi_1^i * phi_2^j,
            where phi_1 = u1 * L^{1/nu} and phi_2 = u2 * L^{-y}.

        Returns
        -------
        np.ndarray
            Normalized residuals (y_i - y_fitted) / d_i.
        """
        w_i=((self.p_i-p_c))
        # w_i=((self.p_i-p_c)/p_c)
        u1_i=(b1)@w_i**np.arange(len(b1))[:,np.newaxis]    # (n_sample,) because b10=0 to ensure u1(w=0)=0
        u2_i=(b2)@w_i**np.arange(len(b2))[:,np.newaxis]  # (n_sample,)
        phi_1=u1_i*(self.L_i)**(1/nu)    # (n_sample,)
        phi_2=u2_i*(self.L_i)**(-y)  # (n_sample,)
        self.phi_1_=phi_1 ** np.arange(a.shape[0])[:,np.newaxis]    # (n1+1,n_sample)
        self.phi_2_=phi_2 ** np.arange(a.shape[1])[:,np.newaxis]    # (n2+1,n_sample)
        self.a=a
        self.y_i_fitted=np.einsum('ij,ik,kj->j',self.phi_1_,self.a,self.phi_2_)

        return (self.y_i-self.y_i_fitted)/self.d_i
    
    def datacollapse(
        self,
        p_c: float | None = None,
        nu: float | None = None,
        beta: float | None = None,
        p_c_vary: bool = True,
        nu_vary: bool = True,
        beta_vary: bool = False,
        nu_range: tuple[float, float] = (0.5, 2),
        p_c_range: tuple[float, float] = (0, 1),
        beta_range: tuple[float, float] = (0, 1),
        **kwargs,
    ) -> MinimizerResult:
        """Perform data collapse optimization without irrelevant corrections.

        Fits critical exponents (p_c, nu, beta) by minimizing deviations from
        a smooth universal scaling curve. The scaling form is:
        y * L^{beta/nu} = f((p - p_c) * L^{1/nu})

        Parameters
        ----------
        p_c : float | None, optional
            Initial guess for critical point.
        nu : float | None, optional
            Initial guess for correlation length exponent.
        beta : float | None, optional
            Initial guess for order parameter exponent.
        p_c_vary : bool, optional
            Whether to fit p_c, by default True.
        nu_vary : bool, optional
            Whether to fit nu, by default True.
        beta_vary : bool, optional
            Whether to fit beta, by default False.
        nu_range : tuple[float, float], optional
            Bounds (min, max) for nu, by default (0.5, 2).
        p_c_range : tuple[float, float], optional
            Bounds (min, max) for p_c, by default (0, 1).
        beta_range : tuple[float, float], optional
            Bounds (min, max) for beta, by default (0, 1).
        **kwargs
            Additional arguments passed to `lmfit.minimize()`.

        Returns
        -------
        MinimizerResult
            The lmfit optimization result. Fitted values are also stored as
            `self.p_c`, `self.nu`, `self.beta`, and `self.res`.
        """
        from lmfit import minimize, Parameters

        params=Parameters()
        params.add('p_c',value=p_c,min=p_c_range[0],max=p_c_range[1],vary=p_c_vary)
        params.add('nu',value=nu,min=nu_range[0],max=nu_range[1],vary=nu_vary)
        params.add('beta',value=beta,min=beta_range[0],max=beta_range[1],vary=beta_vary)
        def residual(params):
            p_c,nu, beta=params['p_c'],params['nu'], params['beta']
            return self.loss(p_c,nu,beta)

        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.beta=res.params['beta'].value
        self.res=res
        self.x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
        self._fit_type='powerlaw'
        return res

    def datacollapse_with_drift(
        self,
        m1: int,
        m2: int,
        n1: int,
        n2: int,
        p_c: float | None = None,
        nu: float | None = None,
        y: float | None = None,
        b1: list[float] | np.ndarray | None = None,
        b2: list[float] | np.ndarray | None = None,
        a: np.ndarray | None = None,
        p_c_vary: bool = True,
        nu_vary: bool = True,
        p_c_range: tuple[float, float] = (0, 1),
        y_vary: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> MinimizerResult:
        """Perform data collapse with irrelevant scaling corrections (Taylor expansion).

        Fits critical exponents and polynomial coefficients using the full
        Taylor expansion approach. All coefficients (b1, b2, a) are fitted.

        Parameters
        ----------
        m1 : int
            Polynomial order for relevant scaling variable coefficients b1.
        m2 : int
            Polynomial order for irrelevant scaling variable coefficients b2.
        n1 : int
            Maximum power of phi_1 in the scaling function expansion.
        n2 : int
            Maximum power of phi_2 in the scaling function expansion.
        p_c : float | None, optional
            Initial guess for critical point.
        nu : float | None, optional
            Initial guess for correlation length exponent.
        y : float | None, optional
            Initial guess for irrelevant scaling exponent.
        b1 : list[float] | np.ndarray | None, optional
            Initial coefficients for relevant scaling. If None, random init.
        b2 : list[float] | np.ndarray | None, optional
            Initial coefficients for irrelevant scaling. If None, random init.
        a : np.ndarray | None, optional
            Initial coefficient matrix. If None, random init.
        p_c_vary : bool, optional
            Whether to fit p_c, by default True.
        nu_vary : bool, optional
            Whether to fit nu, by default True.
        p_c_range : tuple[float, float], optional
            Bounds (min, max) for p_c, by default (0, 1).
        y_vary : bool, optional
            Whether to fit y, by default True.
        seed : int | None, optional
            Random seed for initial coefficient values.
        **kwargs
            Additional arguments passed to `lmfit.minimize()`.

        Returns
        -------
        MinimizerResult
            The lmfit optimization result. Fitted values stored as attributes.
        """
        from lmfit import minimize, Parameters

        params=Parameters()
        params.add('p_c',value=p_c,min=p_c_range[0],max=p_c_range[1],vary=p_c_vary)
        params.add('nu',value=nu,min=0,max=3,vary=nu_vary)
        params.add('y',value=y,min=0,vary=y_vary)
        rng=np.random.default_rng(seed)
        if b1 is None:
            # b1=[0]*(m1+1)
            b1=rng.normal(size=(m1+1))
        if b2 is None:
            # b2=[0]*(m2+1)
            b2=rng.normal(size=(m2+1))
        if a is None:
            # a=np.array([[0]*(n2+1)]*(n1+1))
            a=rng.normal(size=(n1+1,n2+1))
        for i in range(m1+1):
            if i == 0:
                params.add(f'b_1_{i}',value=0,vary=False)
            else:
                params.add(f'b_1_{i}',value=b1[i])
        for i in range(m2+1):
            params.add(f'b_2_{i}',value=b2[i])
        for i in range(n1+1):
            for j in range(n2+1):
                if (i==1 and j==0) or (i==0 and j==1):
                    params.add(f'a_{i}_{j}',value=1,vary=False)
                else:
                    params.add(f'a_{i}_{j}',value=a[i,j])
        def residual(params):
            return self.loss_with_drift(params['p_c'],params['nu'],params['y'],[params[f'b_1_{i}'] for i in range(m1+1)],[params[f'b_2_{i}'] for i in range(m2+1)],np.array([[params[f'a_{i}_{j}'] for j in range(n2+1)] for i in range(n1+1)]))
        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.y=res.params['y'].value
        self.res=res

        self.x_i=(self.p_i-self.p_c)*(self.L_i)**(1/nu)

        self.y_i_minus_irrelevant=self.y_i-np.einsum('ij,ik,kj->j',self.phi_1_,self.a[:,1:],self.phi_2_[1:,:])
        return res
    
    def loss_with_drift_GLS(
        self,
        p_c: float,
        nu: float,
        y: float,
        n1: int,
        n2: int,
        beta: float = 0,
    ) -> np.ndarray:
        """Compute residuals using generalized least squares (GLS).

        Fits the scaling form with finite-size corrections:

            y(p, L) * L^{beta/nu} = sum_{j1=0}^{n1} sum_{j2=0}^{n2} a_{j1,j2} * x^{j1} * L^{-y*j2}

        where:
            - x = (p - p_c) * L^{1/nu} : relevant scaling variable
            - L^{-y} : irrelevant scaling variable (vanishes as L -> inf)
            - a_{j1,j2} : polynomial coefficients (solved via GLS)

        The observable decomposes into:
            - Relevant part (universal): f(x) = sum_{j1} a_{j1,0} * x^{j1}
            - Irrelevant part (corrections): sum_{j1,j2>0} a_{j1,j2} * x^{j1} * L^{-y*j2}

        Nonlinear parameters (p_c, nu, y) are optimized via lmfit; linear
        coefficients a_{j1,j2} are solved analytically by GLS.

        Parameters
        ----------
        p_c : float
            Critical point of the tuning parameter.
        nu : float
            Correlation length exponent.
        y : float
            Irrelevant scaling exponent.
        n1 : int
            Maximum power of the relevant scaling variable x.
        n2 : int
            Maximum power of the irrelevant scaling variable L^{-y}.
        beta : float, optional
            Order parameter exponent, by default 0.

        Returns
        -------
        np.ndarray
            Normalized residuals (y_scaled - y_fitted) / d_scaled.
        """
        x_i=(self.p_i-p_c)*(self.L_i)**(1/nu)
        ir_i=self.L_i**(-y)
        Y=self.y_i * self.L_i**(beta/nu)
        d_scaled=self.d_i * self.L_i**(beta/nu)
        return self._gls_solve(x_i, Y, d_scaled, ir_i, n1, n2)

    
    def datacollapse_with_drift_GLS(
        self,
        n1: int,
        n2: int,
        p_c: float | None = None,
        nu: float | None = None,
        y: float | None = None,
        beta: float = 0,
        p_c_range: tuple[float, float] = (0, 1),
        nu_range: tuple[float, float] = (0, 2),
        y_range: tuple[float, float] = (0, 5),
        beta_range: tuple[float, float] = (0, 2),
        p_c_vary: bool = True,
        nu_vary: bool = True,
        y_vary: bool = True,
        beta_vary: bool = False,
        **kwargs,
    ) -> MinimizerResult:
        """Perform data collapse with GLS fitting for polynomial coefficients.

        Fits critical exponents (p_c, nu, y, beta) while using generalized least
        squares to solve for optimal polynomial coefficients. More efficient than
        `datacollapse_with_drift` when the scaling function is well-approximated
        by a polynomial.

        The scaling form is:
        y(p, L) = L^{-beta/nu} * sum_{j1,j2} a_{j1,j2} * x^{j1} * L^{-y*j2}
        where x = (p - p_c) * L^{1/nu}

        Parameters
        ----------
        n1 : int
            Maximum power of the relevant scaling variable x.
        n2 : int
            Maximum power of the irrelevant scaling variable L^{-y}.
        p_c : float | None, optional
            Initial guess for critical point.
        nu : float | None, optional
            Initial guess for correlation length exponent.
        y : float | None, optional
            Initial guess for irrelevant scaling exponent.
        beta : float, optional
            Initial guess for order parameter exponent, by default 0.
        p_c_range : tuple[float, float], optional
            Bounds (min, max) for p_c, by default (0, 1).
        nu_range : tuple[float, float], optional
            Bounds (min, max) for nu, by default (0, 2).
        y_range : tuple[float, float], optional
            Bounds (min, max) for y, by default (0, 5).
        beta_range : tuple[float, float], optional
            Bounds (min, max) for beta, by default (0, 2).
        p_c_vary : bool, optional
            Whether to fit p_c, by default True.
        nu_vary : bool, optional
            Whether to fit nu, by default True.
        y_vary : bool, optional
            Whether to fit y, by default True.
        beta_vary : bool, optional
            Whether to fit beta, by default False.
        **kwargs
            Additional arguments passed to `lmfit.minimize()`.

        Returns
        -------
        MinimizerResult
            The lmfit optimization result. Fitted values and polynomial
            coefficients stored as attributes.
        """
        from lmfit import minimize, Parameters

        params=Parameters()
        params.add('p_c',value=p_c,min=p_c_range[0],max=p_c_range[1],vary=p_c_vary)
        params.add('nu',value=nu,min=nu_range[0],max=nu_range[1],vary=nu_vary)
        params.add('y',value=y,min=y_range[0],max=y_range[1],vary=y_vary)
        params.add('beta',value=beta,min=beta_range[0],max=beta_range[1],vary=beta_vary)

        def residual(params):
            return self.loss_with_drift_GLS(params['p_c'],params['nu'],params['y'],n1,n2,params['beta'])
        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.nu=res.params['nu'].value
        self.y=res.params['y'].value
        self.beta=res.params['beta'].value
        self.res=res
        self.x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
        if n2>0:
            self.y_i_minus_irrelevant=self.y_i_scaled - self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.coeffs.reshape((n1+1,n2+1))[:,1:].flatten()
            self.y_i_irrelevant=self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.coeffs.reshape((n1+1,n2+1))[:,1:].flatten()
        else:
            self.y_i_minus_irrelevant=self.y_i_scaled
            self.y_i_irrelevant=0
        self._fit_type='powerlaw'
        return res

    def loss_bkt(self, p_c, L_0, sigma, delta=0):
        """BKT scaling residuals: O ~ L^delta * f((p-p_c) * (log(L/L_0))^{1/sigma})"""
        if np.any(self.L_i <= L_0):
            raise ValueError(f'L_0={L_0} must be less than all system sizes L_i (min={np.min(self.L_i)})')
        x_i = (self.p_i - p_c) * (np.log(self.L_i / L_0))**(1/sigma)
        y_scaled = self.y_i * self.L_i**(-delta)
        d_scaled = self.d_i * self.L_i**(-delta)
        return self._smoothness_residuals(x_i, y_scaled, d_scaled)

    def datacollapse_bkt(
        self,
        p_c: float | None = None,
        L_0: float | None = None,
        sigma: float | None = None,
        delta: float = 0,
        p_c_vary: bool = True,
        L_0_vary: bool = True,
        sigma_vary: bool = True,
        delta_vary: bool = False,
        p_c_range: tuple[float, float] = (0, 1),
        L_0_range: tuple[float, float] | None = None,
        sigma_range: tuple[float, float] = (0.1, 5),
        delta_range: tuple[float, float] = (-2, 2),
        **kwargs,
    ) -> MinimizerResult:
        """Perform BKT data collapse optimization without irrelevant corrections.

        Fits critical parameters (p_c, L_0, sigma, delta) by minimizing deviations
        from a smooth universal scaling curve. The BKT scaling form is:
        y * L^{-delta} = f((p - p_c) * (log(L/L_0))^{1/sigma})

        Parameters
        ----------
        p_c : float | None, optional
            Initial guess for critical point.
        L_0 : float | None, optional
            Initial guess for the BKT length scale.
        sigma : float | None, optional
            Initial guess for the BKT exponent.
        delta : float, optional
            Initial guess for the scaling exponent of y, by default 0.
        p_c_vary : bool, optional
            Whether to fit p_c, by default True.
        L_0_vary : bool, optional
            Whether to fit L_0, by default True.
        sigma_vary : bool, optional
            Whether to fit sigma, by default True.
        delta_vary : bool, optional
            Whether to fit delta, by default False.
        p_c_range : tuple[float, float], optional
            Bounds (min, max) for p_c, by default (0, 1).
        L_0_range : tuple[float, float] | None, optional
            Bounds (min, max) for L_0. If None, auto-computed as (0.01, min(L_i)-eps).
        sigma_range : tuple[float, float], optional
            Bounds (min, max) for sigma, by default (0.1, 5).
        delta_range : tuple[float, float], optional
            Bounds (min, max) for delta, by default (-2, 2).
        **kwargs
            Additional arguments passed to `lmfit.minimize()`.

        Returns
        -------
        MinimizerResult
            The lmfit optimization result. Fitted values are also stored as
            `self.p_c`, `self.L_0`, `self.sigma`, `self.delta`, and `self.res`.
        """
        from lmfit import minimize, Parameters

        if L_0_range is None:
            L_0_range = (0.01, float(np.min(self.L_i) - 1e-6))

        params=Parameters()
        params.add('p_c',value=p_c,min=p_c_range[0],max=p_c_range[1],vary=p_c_vary)
        params.add('L_0',value=L_0,min=L_0_range[0],max=L_0_range[1],vary=L_0_vary)
        params.add('sigma',value=sigma,min=sigma_range[0],max=sigma_range[1],vary=sigma_vary)
        params.add('delta',value=delta,min=delta_range[0],max=delta_range[1],vary=delta_vary)

        def residual(params):
            r = self.loss_bkt(params['p_c'],params['L_0'],params['sigma'],params['delta'])
            return r[np.isfinite(r)]

        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.L_0=res.params['L_0'].value
        self.sigma=res.params['sigma'].value
        self.delta=res.params['delta'].value
        self.res=res
        self.x_i=(self.p_i-self.p_c)*(np.log(self.L_i/self.L_0))**(1/self.sigma)
        self._fit_type='bkt'
        return res

    def loss_bkt_with_drift_GLS(self, p_c, L_0, sigma, y, n1, n2, delta=0):
        """BKT scaling GLS residuals with irrelevant corrections.

        Note: BKT GLS is experimental and not yet verified against analytical results.
        """
        if np.any(self.L_i <= L_0):
            raise ValueError(f'L_0={L_0} must be less than all system sizes L_i (min={np.min(self.L_i)})')
        x_i = (self.p_i - p_c) * (np.log(self.L_i / L_0))**(1/sigma)
        ir_i = self.L_i**(-y)
        Y = self.y_i * self.L_i**(-delta)
        d_scaled = self.d_i * self.L_i**(-delta)
        return self._gls_solve(x_i, Y, d_scaled, ir_i, n1, n2)

    def datacollapse_bkt_with_drift_GLS(
        self,
        n1: int,
        n2: int,
        p_c: float | None = None,
        L_0: float | None = None,
        sigma: float | None = None,
        y: float | None = None,
        delta: float = 0,
        p_c_vary: bool = True,
        L_0_vary: bool = True,
        sigma_vary: bool = True,
        y_vary: bool = True,
        delta_vary: bool = False,
        p_c_range: tuple[float, float] = (0, 1),
        L_0_range: tuple[float, float] | None = None,
        sigma_range: tuple[float, float] = (0.1, 5),
        y_range: tuple[float, float] = (0, 5),
        delta_range: tuple[float, float] = (-2, 2),
        **kwargs,
    ) -> MinimizerResult:
        """BKT data collapse with GLS fitting.

        Note: BKT GLS is experimental and not yet verified against analytical results.

        Fits critical parameters (p_c, L_0, sigma, y, delta) while using generalized
        least squares to solve for optimal polynomial coefficients. The BKT scaling
        form with irrelevant corrections is:
        y(p, L) * L^{-delta} = sum_{j1,j2} a_{j1,j2} * x^{j1} * L^{-y*j2}
        where x = (p - p_c) * (log(L/L_0))^{1/sigma}

        Parameters
        ----------
        n1 : int
            Maximum power of the relevant scaling variable x.
        n2 : int
            Maximum power of the irrelevant scaling variable L^{-y}.
        p_c : float | None, optional
            Initial guess for critical point.
        L_0 : float | None, optional
            Initial guess for the BKT length scale.
        sigma : float | None, optional
            Initial guess for the BKT exponent.
        y : float | None, optional
            Initial guess for irrelevant scaling exponent.
        delta : float, optional
            Initial guess for the scaling exponent of y, by default 0.
        p_c_vary : bool, optional
            Whether to fit p_c, by default True.
        L_0_vary : bool, optional
            Whether to fit L_0, by default True.
        sigma_vary : bool, optional
            Whether to fit sigma, by default True.
        y_vary : bool, optional
            Whether to fit y, by default True.
        delta_vary : bool, optional
            Whether to fit delta, by default False.
        p_c_range : tuple[float, float], optional
            Bounds (min, max) for p_c, by default (0, 1).
        L_0_range : tuple[float, float] | None, optional
            Bounds (min, max) for L_0. If None, auto-computed as (0.01, min(L_i)-eps).
        sigma_range : tuple[float, float], optional
            Bounds (min, max) for sigma, by default (0.1, 5).
        y_range : tuple[float, float], optional
            Bounds (min, max) for y, by default (0, 5).
        delta_range : tuple[float, float], optional
            Bounds (min, max) for delta, by default (-2, 2).
        **kwargs
            Additional arguments passed to `lmfit.minimize()`.

        Returns
        -------
        MinimizerResult
            The lmfit optimization result. Fitted values and polynomial
            coefficients stored as attributes.
        """
        from lmfit import minimize, Parameters

        if L_0_range is None:
            L_0_range = (0.01, float(np.min(self.L_i) - 1e-6))

        params=Parameters()
        params.add('p_c',value=p_c,min=p_c_range[0],max=p_c_range[1],vary=p_c_vary)
        params.add('L_0',value=L_0,min=L_0_range[0],max=L_0_range[1],vary=L_0_vary)
        params.add('sigma',value=sigma,min=sigma_range[0],max=sigma_range[1],vary=sigma_vary)
        params.add('y',value=y,min=y_range[0],max=y_range[1],vary=y_vary)
        params.add('delta',value=delta,min=delta_range[0],max=delta_range[1],vary=delta_vary)

        def residual(params):
            r = self.loss_bkt_with_drift_GLS(params['p_c'],params['L_0'],params['sigma'],params['y'],n1,n2,params['delta'])
            return r[np.isfinite(r)]

        res=minimize(residual,params,**kwargs)
        self.p_c=res.params['p_c'].value
        self.L_0=res.params['L_0'].value
        self.sigma=res.params['sigma'].value
        self.y=res.params['y'].value
        self.delta=res.params['delta'].value
        self.res=res
        self.x_i=(self.p_i-self.p_c)*(np.log(self.L_i/self.L_0))**(1/self.sigma)
        if n2>0:
            self.y_i_minus_irrelevant=self.y_i_scaled - self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.coeffs.reshape((n1+1,n2+1))[:,1:].flatten()
            self.y_i_irrelevant=self.X.reshape((-1,n1+1,n2+1))[:,:,1:].reshape((-1,(n1+1)*n2))@self.coeffs.reshape((n1+1,n2+1))[:,1:].flatten()
        else:
            self.y_i_minus_irrelevant=self.y_i_scaled
            self.y_i_irrelevant=0
        self._fit_type='bkt'
        return res
        



    def plot_data_collapse(
        self,
        ax: Axes | None = None,
        drift: bool = False,
        driftcollapse: bool = False,
        plot_irrelevant: bool = True,
        errorbar: bool = False,
        abs: bool = False,
        color_iter: Iterator[Any] | None = None,
        raw: bool = False,
        plot_kind: str = "scatter",
        plot_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Plot the data collapse results.

        Visualizes the collapsed data with different options for showing
        raw data, drift corrections, and irrelevant contributions.

        Parameters
        ----------
        ax : Axes | None, optional
            Matplotlib axes to plot on. If None, creates new figure.
        drift : bool, optional
            If True, plot results from drift-corrected fitting.
        driftcollapse : bool, optional
            If True with drift, plot collapsed data minus irrelevant part.
            If False with drift, plot raw data with fitted curve.
        plot_irrelevant : bool, optional
            If True with driftcollapse, plot irrelevant contribution on twin axis.
        errorbar : bool, optional
            If True, show error bars on data points.
        abs : bool, optional
            If True, use absolute value of x-axis variable.
        color_iter : Iterator[Any] | None, optional
            Custom color iterator for different system sizes.
        raw : bool, optional
            If True, plot raw (unscaled) data.
        plot_kind : str, optional
            The kind of plot to create. Options are 'scatter' (default) or 'line'.
        plot_kwargs : dict[str, Any] | None, optional
            Additional keyword arguments to pass to the plotting function
            (ax.scatter or ax.plot).
        **kwargs
            Additional arguments passed to scatter/plot functions.
        """
        def _wrap_math(label: str, is_math: bool) -> str:
            return f"${label}$" if is_math else label

        def _title_from_params(params: list[tuple[str, float, float | None]]) -> str:
            if all(stderr is not None for _, _, stderr in params):
                parts = [f"{label}={value:.3f}$\\pm${stderr:.3f}" for label, value, stderr in params]
            else:
                parts = [f"{label}={value:.3f}" for label, value, _ in params]
            return ",".join(parts)

        if plot_kwargs is None:
            plot_kwargs = {}
        
        final_plot_kwargs = {**plot_kwargs, **kwargs}

        if raw:
            x_i=self.p_i
            y_i=self.y_i
            d_i=self.d_i
        else:
            if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                x_i = self.x_i
                y_i = self.y_i * self.L_i**(-self.delta)
                d_i = self.d_i * self.L_i**(-self.delta)
            else:
                x_i=(self.p_i-self.p_c)*(self.L_i)**(1/self.nu)
                y_i= self.y_i*self.L_i**(self.beta/self.nu)
                d_i = self.d_i * self.L_i**(self.beta/self.nu)
        # x_i=self.p_i
        if ax is None:
            fig,ax = plt.subplots()
        L_list=self.df.index.get_level_values(self.L_).unique().sort_values().values
        idx_list=[0]+(np.cumsum([self.df.xs(key=L,level=self.L_).shape[0] for L in L_list])).tolist()
        L_dict={L:(start_idx,end_idx) for L,start_idx,end_idx in zip(L_list,idx_list[:-1],idx_list[1:])}
        # color_iter=iter(plt.cm.rainbow(np.linspace(0,1,len(L_list))))
        if color_iter is None:
            color_iter = iter(plt.cm.Blues(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
        color_r_iter = iter(plt.cm.Reds(0.4+0.6*(i/L_list.shape[0])) for i in range(L_list.shape[0]))
        if drift and driftcollapse and plot_irrelevant:
            ax2=ax.twinx()
            if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                l_label, l_is_math = _format_exponent(self.L_, r"-\Delta")
            else:
                l_label, l_is_math = _format_exponent(self.L_, r"\beta/\nu")
            if hasattr(self, "y_i_scaled"):
                if l_is_math:
                    ax2.set_ylabel(rf"$y_{{irre}} {l_label}$")
                else:
                    ax2.set_ylabel(rf"$y_{{irre}}$ {l_label}")
            else:
                ax2.set_ylabel(r"$y_{irre}$")
        for L,(start_idx,end_idx) in L_dict.items():
            color=next(color_iter)
            if drift:
                if not driftcollapse:
                    if abs:
                        x=np.abs(self.p_i[start_idx:end_idx])
                    else:
                        x=(self.p_i[start_idx:end_idx])
                    ax.errorbar(x, self.y_i[start_idx:end_idx], label=f'{L}', color=color, yerr=self.d_i[start_idx:end_idx], capsize=2, fmt='x',linestyle="None")

                    ax.plot(x,self.y_i_fitted[start_idx:end_idx],label=f'{L}',color=color,**final_plot_kwargs)
                else:
                    if abs:
                        x=np.abs(x_i[start_idx:end_idx])
                    else:
                        x=(x_i[start_idx:end_idx])
                    if plot_kind == "line":
                        ax.plot(x,self.y_i_minus_irrelevant[start_idx:end_idx],label=f'{L}',color=color,**final_plot_kwargs)
                    else:
                        ax.scatter(x,self.y_i_minus_irrelevant[start_idx:end_idx],label=f'{L}',color=color,**final_plot_kwargs)
                    if plot_irrelevant:
                        color_r=next(color_r_iter)
                        if plot_kind == "line":
                            ax2.plot(x,self.y_i_irrelevant[start_idx:end_idx],label=f'{L}',color=color_r,**final_plot_kwargs)
                        else:
                            ax2.scatter(x,self.y_i_irrelevant[start_idx:end_idx],label=f'{L}',color=color_r,**final_plot_kwargs)
            else:
                if abs:
                    x=np.abs(x_i[start_idx:end_idx])
                else:
                    x=x_i[start_idx:end_idx]
                if errorbar:
                    ax.errorbar(x,y_i[start_idx:end_idx],yerr=d_i[start_idx:end_idx],label=f'{L}',color=color,capsize=3,**final_plot_kwargs)
                else:
                    if plot_kind == "line":
                        ax.plot(x,y_i[start_idx:end_idx],label=f'{L}',color=color,**final_plot_kwargs)
                    else:
                        ax.scatter(x,y_i[start_idx:end_idx],label=f'{L}',color=color,**final_plot_kwargs)
        if raw:
            ax.set_ylabel(r"$y_i$")
        else:
            if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                l_label, l_is_math = _format_exponent(self.L_, r"-\Delta")
            else:
                l_label, l_is_math = _format_exponent(self.L_, r"\beta/\nu")
            if l_is_math:
                ax.set_ylabel(rf"$y_i {l_label}$")
            else:
                ax.set_ylabel(rf"$y_i$ {l_label}")
        if raw:
            ax.set_xlabel(_append_subscript(self.p_, "i"))
        else:
            if drift:
                if not driftcollapse:
                    ax.set_xlabel(_append_subscript(self.p_, "i"))
                    ax.set_ylabel(r"$y_i$")
                    if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                        ax.set_title(
                            _title_from_params(
                                [
                                    (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                    (r"$\sigma$", self.sigma, self.res.params["sigma"].stderr),
                                    (r"$\Delta$", self.delta, self.res.params["delta"].stderr),
                                    (r"$L_0$", self.L_0, self.res.params["L_0"].stderr),
                                    (r"$y$", self.y, self.res.params["y"].stderr),
                                ]
                            )
                        )
                    else:
                        ax.set_title(
                            _title_from_params(
                                [
                                    (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                    (r"$\nu$", self.nu, self.res.params["nu"].stderr),
                                    (r"$y$", self.y, self.res.params["y"].stderr),
                                ]
                            )
                        )

                else:
                    ax.set_xlabel(r'$x_i$')
                    if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                        ax.set_title(
                            _title_from_params(
                                [
                                    (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                    (r"$\sigma$", self.sigma, self.res.params["sigma"].stderr),
                                    (r"$\Delta$", self.delta, self.res.params["delta"].stderr),
                                    (r"$L_0$", self.L_0, self.res.params["L_0"].stderr),
                                    (r"$y$", self.y, self.res.params["y"].stderr),
                                ]
                            )
                        )
                    else:
                        ax.set_title(
                            _title_from_params(
                                [
                                    (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                    (r"$\nu$", self.nu, self.res.params["nu"].stderr),
                                    (r"$y$", self.y, self.res.params["y"].stderr),
                                ]
                            )
                        )
                    if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                        l_label, l_is_math = _format_exponent(self.L_, r"-\Delta")
                    else:
                        l_label, l_is_math = _format_exponent(self.L_, r"\beta/\nu")
                    if hasattr(self, "y_i_scaled"):
                        if l_is_math:
                            ax.set_ylabel(rf"$(y_i - y_{{irre}}) {l_label}$")
                        else:
                            ax.set_ylabel(rf"$(y_i - y_{{irre}})$ {l_label}")
                    else:
                        ax.set_ylabel(r"$y_i - y_{irre}$")
            else:
                if getattr(self, '_fit_type', 'powerlaw') == 'bkt':
                    p_safe = _math_safe(self.p_)
                    l_safe = _math_safe(self.L_)
                    ax.set_xlabel(r'$(' + p_safe + r'_i - ' + p_safe + r'_c) (\log(' + l_safe + r'/L_0))^{1/\sigma}$')
                    ax.set_title(
                        _title_from_params(
                            [
                                (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                (r"$\sigma$", self.sigma, self.res.params["sigma"].stderr),
                                (r"$\Delta$", self.delta, self.res.params["delta"].stderr),
                                (r"$L_0$", self.L_0, self.res.params["L_0"].stderr),
                            ]
                        )
                    )
                else:
                    l_label, l_is_math = _format_exponent(self.L_, r"1/\nu")
                    l_label = _wrap_math(l_label, l_is_math)
                    if abs:
                        ax.set_xlabel(
                            f"|{_append_subscript(self.p_, 'i')}-{_append_subscript(self.p_, 'c')}| {l_label}"
                        )
                    else:
                        ax.set_xlabel(
                            f"({_append_subscript(self.p_, 'i')}-{_append_subscript(self.p_, 'c')}) {l_label}"
                        )
                    # ax.set_title(rf'$p_c={self.p_c:.3f},\nu={self.nu:.3f}$')
                    ax.set_title(
                        _title_from_params(
                            [
                                (_append_subscript(self.p_, "c"), self.p_c, self.res.params["p_c"].stderr),
                                (r"$\nu$", self.nu, self.res.params["nu"].stderr),
                                (r"$\beta$", self.beta, self.res.params["beta"].stderr),
                            ]
                        )
                    )

        ax.legend()
        ax.grid('on')

        # adder=self.df.index.get_level_values('adder').unique().tolist()[0]
        # print(f'{self.params["Metrics"]}_Scaling_L({L_list[0]},{L_list[-1]})_adder({adder[0]}-{adder[1]}).png')

    def parameter_sweep(
        self,
        p_c: np.ndarray | list[float] | float,
        nu: np.ndarray | list[float] | float,
        beta: np.ndarray | list[float] | float = 0,
        p_c_range: tuple[float, float] | None = None,
        nu_range: tuple[float, float] | None = None,
        beta_range: tuple[float, float] | None = None,
        ax: Axes | None = None,
        colorbar_position: str = 'right',
        n_jobs: int = -1,
        backend: str = 'threading',
        cmap: str = 'seismic',
        log_chi2: bool = True,
    ) -> dict[str, Any]:
        """Sweep over two parameters to visualize reduced chi-squared landscape.

        Computes reduced chi-squared for a 2D grid of parameter values, displays
        a pcolormesh plot, and extracts optimal values with error bounds from
        a contour at 1.3× the minimum chi-squared.

        Note: This method only supports the basic data collapse (without scaling
        corrections). 

        **Convention:** Each parameter (p_c, nu, beta) can be:

        - A list or array → swept over (used as grid axis)
        - A scalar → held fixed at that value

        Exactly 2 parameters must be arrays; the third must be a scalar.

        Parameters
        ----------
        p_c : array-like or float
            Critical point values. If array, swept over; if scalar, held fixed.
        nu : array-like or float
            Correlation length exponent values. If array, swept; if scalar, fixed.
        beta : array-like or float, optional
            Order parameter exponent values. Default is 0 (scalar, fixed).
        p_c_range : tuple[float, float] or None, optional
            Bounds for p_c during internal fitting calls.
        nu_range : tuple[float, float], optional
            Bounds for nu during internal fitting calls, by default (0.5, 2).
        beta_range : tuple[float, float], optional
            Bounds for beta during internal fitting calls, by default (0, 1).
        ax : Axes or None, optional
            Matplotlib axes. If None, creates new figure.
        colorbar_position : str, optional
            Position of colorbar: 'right' or 'top', by default 'right'.
        n_jobs : int, optional
            Number of parallel jobs. Default -1 uses all CPU cores.
            Set to 1 for serial execution (no parallelization).
        backend : str, optional
            Joblib backend: 'loky' (multiprocessing, default) or 'threading'.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - 'p_c', 'nu', 'beta': optimal values at minimum chi-squared
            - 'p_c_error', 'nu_error', 'beta_error': (min, max) tuples from contour
              (None for the fixed parameter)
            - 'chi2_grid': the computed chi-squared array

        Raises
        ------
        NotImplementedError
            If not exactly 2 parameters are arrays.

        Examples
        --------
        >>> # Sweep p_c and nu, fix beta=0
        >>> result = dc.parameter_sweep(
        ...     p_c=np.linspace(0.4, 0.6, 20),
        ...     nu=np.linspace(0.8, 1.5, 20),
        ...     beta=0,
        ... )

        >>> # Sweep nu and beta, fix p_c=0.5
        >>> result = dc.parameter_sweep(
        ...     p_c=0.5,
        ...     nu=np.linspace(0.8, 1.5, 20),
        ...     beta=np.linspace(0, 0.5, 15),
        ... )
        """
        import matplotlib.pyplot as plt

        def is_iterable(x):
            return isinstance(x, (list, np.ndarray))

        # Classify parameters as sweep or fixed, preserving order
        param_info = [
            ('p_c', p_c, self.p_),
            ('nu', nu, r'$\nu$'),
            ('beta', beta, r'$\beta$'),
        ]
        sweep_params = []
        fixed_params = {}
        for name, val, label in param_info:
            if is_iterable(val):
                sweep_params.append((name, np.asarray(val), label))
            else:
                fixed_params[name] = val

        if len(sweep_params) != 2:
            raise NotImplementedError(
                f"Exactly 2 parameter arrays required for pcolormesh, got {len(sweep_params)}. "
                "Pass arrays for 2 parameters and a scalar for the third."
            )

        # Extract sweep arrays (order preserved from signature)
        name1, arr1, label1 = sweep_params[0]
        name2, arr2, label2 = sweep_params[1]

        # Auto-derive ranges from sweep arrays so lmfit doesn't clamp values
        def _auto_range(val, default):
            if isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                return (float(arr.min()), float(arr.max()))
            return default

        if p_c_range is None:
            p_c_range = _auto_range(p_c, (0, 1))
        if nu_range is None:
            nu_range = _auto_range(nu, (0., 10))
        if beta_range is None:
            beta_range = _auto_range(beta, (0, 10))

        # Define chi2 computation function
        def compute_chi2(i, j):
            params = dict(fixed_params)
            params[name1] = arr1[i]
            params[name2] = arr2[j]
            try:
                res = self.datacollapse(
                    p_c=params['p_c'], nu=params['nu'], beta=params['beta'],
                    p_c_vary=False, nu_vary=False, beta_vary=False,
                    p_c_range=p_c_range,
                    nu_range=nu_range, beta_range=beta_range,
                )
                return i, j, res.redchi
            except Exception:
                return i, j, np.nan

        # Generate all tasks
        tasks = [(i, j) for i in range(len(arr1)) for j in range(len(arr2))]

        # Execute
        if n_jobs == 1:
            results = [compute_chi2(i, j) for i, j in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(compute_chi2)(i, j) for i, j in tasks
            )

        # Assemble grid
        chi2_grid = np.zeros((len(arr1), len(arr2)))
        for i, j, val in results:
            chi2_grid[i, j] = val

        def _trim_nan_edges(values: np.ndarray, grid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
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

        plot_arr1, plot_grid = _trim_nan_edges(arr1, chi2_grid, axis=0)
        plot_arr2, plot_grid = _trim_nan_edges(arr2, plot_grid, axis=1)

        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        def _edges_from_centers(values: np.ndarray) -> np.ndarray:
            if values.size == 1:
                return np.array([values[0] - 0.5, values[0] + 0.5])
            deltas = np.diff(values)
            first = values[0] - deltas[0] / 2
            last = values[-1] + deltas[-1] / 2
            mids = values[:-1] + deltas / 2
            return np.concatenate([[first], mids, [last]])

        # Plot pcolormesh
        im = ax.pcolormesh(
            plot_arr2, plot_arr1, np.log(plot_grid) if log_chi2 else plot_grid,
            cmap=cmap,
            shading='auto',
            # vmax=np.log(2.6 * np.nanmin(chi2_grid))
        )
        ax.margins(x=0, y=0)
        x_edges = _edges_from_centers(plot_arr2)
        y_edges = _edges_from_centers(plot_arr1)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        # Set axis labels (first sweep param → ylabel, second → xlabel)
        ax.set_ylabel(_format_token(label1))
        ax.set_xlabel(_format_token(label2))

        # Add colorbar
        cb_label = r'$\log(\chi_\nu^2)$' if log_chi2 else r'$\chi_\nu^2$'
        if colorbar_position == 'right':
            plt.colorbar(im, ax=ax, label=cb_label)
        elif colorbar_position == 'top':
            axins = ax.inset_axes([0.4, 1.02, 0.6, 0.05])
            plt.colorbar(im, cax=axins, label='', orientation='horizontal')
            axins.xaxis.tick_top()
            axins.text(0.4, 1.02, cb_label, ha='right', va='bottom', transform=ax.transAxes)

        # Find minimum and plot marker
        idx1, idx2 = np.unravel_index(np.nanargmin(plot_grid), plot_grid.shape)
        ax.plot(plot_arr2[idx2], plot_arr1[idx1], marker='x', color='red', markersize=10)
        ax.set_title(f'min $\\chi^2_\\nu$ = {plot_grid[idx1, idx2]:.1f}')

        # Draw contour at 1.3× minimum
        pts = ax.contour(
            plot_arr2, plot_arr1, plot_grid,
            levels=[1.3 * plot_grid[idx1, idx2]],
            colors='y'
        )

        # Re-apply bounds to avoid autoscale margins after contour/marker
        ax.margins(x=0, y=0)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_autoscale_on(False)

        # Select contour with most vertices (longest)
        paths = pts.get_paths()
        if paths:
            longest_path = max(paths, key=lambda p: len(p.vertices))
            vertices = longest_path.vertices
            error1 = (vertices[:, 1].min(), vertices[:, 1].max())
            error2 = (vertices[:, 0].min(), vertices[:, 0].max())
        else:
            error1 = (np.nan, np.nan)
            error2 = (np.nan, np.nan)

        # Build result dict
        result = {
            'p_c': None,
            'nu': None,
            'beta': None,
            'p_c_error': None,
            'nu_error': None,
            'beta_error': None,
            'chi2_grid': chi2_grid,
        }

        # Set optimal values
        result[name1] = plot_arr1[idx1]
        result[name2] = plot_arr2[idx2]
        for fname, fval in fixed_params.items():
            result[fname] = fval

        # Set errors for swept params
        result[f'{name1}_error'] = error1
        result[f'{name2}_error'] = error2

        # Print summary
        print(f'{name1}={arr1[idx1]:.4f}, {name2}={arr2[idx2]:.4f}, chi2={chi2_grid[idx1, idx2]:.4f}')
        print(f'{name1} error=({error1[0]:.4f}, {error1[1]:.4f}), {name2} error=({error2[0]:.4f}, {error2[1]:.4f})')

        return result

    def parameter_sweep_bkt(
        self,
        p_c: np.ndarray | list[float] | float,
        sigma: np.ndarray | list[float] | float,
        L_0: np.ndarray | list[float] | float,
        delta: np.ndarray | list[float] | float = 0,
        p_c_range: tuple[float, float] | None = None,
        sigma_range: tuple[float, float] | None = None,
        L_0_range: tuple[float, float] | None = None,
        delta_range: tuple[float, float] | None = None,
        ax: Axes | None = None,
        colorbar_position: str = 'right',
        n_jobs: int = -1,
        backend: str = 'threading',
        cmap: str = 'seismic',
        log_chi2: bool = True,
    ) -> dict[str, Any]:
        """Sweep over two BKT parameters to visualize reduced chi-squared landscape.

        Computes reduced chi-squared for a 2D grid of BKT parameter values, displays
        a pcolormesh plot, and extracts optimal values with error bounds from
        a contour at 1.3× the minimum chi-squared.

        Note: This method only supports the basic BKT data collapse (without scaling
        corrections). Use with BKT scaling only.

        **Convention:** Each parameter (p_c, sigma, L_0, delta) can be:

        - A list or array → swept over (used as grid axis)
        - A scalar → held fixed at that value

        Exactly 2 parameters must be arrays; the other 2 must be scalars.

        Parameters
        ----------
        p_c : array-like or float
            Critical point values. If array, swept over; if scalar, held fixed.
        sigma : array-like or float
            BKT exponent values. If array, swept; if scalar, fixed.
        L_0 : array-like or float
            BKT length scale values. If array, swept; if scalar, fixed.
        delta : array-like or float, optional
            Scaling exponent of y values. Default is 0 (scalar, fixed).
        p_c_range : tuple[float, float] or None, optional
            Bounds for p_c during internal fitting calls.
        sigma_range : tuple[float, float] or None, optional
            Bounds for sigma during internal fitting calls, by default (0.1, 5).
        L_0_range : tuple[float, float] or None, optional
            Bounds for L_0 during internal fitting calls.
        delta_range : tuple[float, float] or None, optional
            Bounds for delta during internal fitting calls, by default (-2, 2).
        ax : Axes or None, optional
            Matplotlib axes. If None, creates new figure.
        colorbar_position : str, optional
            Position of colorbar: 'right' or 'top', by default 'right'.
        n_jobs : int, optional
            Number of parallel jobs. Default -1 uses all CPU cores.
            Set to 1 for serial execution (no parallelization).
        backend : str, optional
            Joblib backend: 'loky' (multiprocessing, default) or 'threading'.
        cmap : str, optional
            Colormap for pcolormesh, by default 'seismic'.
        log_chi2 : bool, optional
            If True, plot log(chi2), by default True.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - 'p_c', 'sigma', 'L_0', 'delta': optimal values at minimum chi-squared
            - 'p_c_error', 'sigma_error', 'L_0_error', 'delta_error': (min, max) tuples from contour
              (None for the fixed parameter)
            - 'chi2_grid': the computed chi-squared array

        Raises
        ------
        NotImplementedError
            If not exactly 2 parameters are arrays.

        Examples
        --------
        >>> # Sweep p_c and sigma, fix L_0 and delta
        >>> result = dc.parameter_sweep_bkt(
        ...     p_c=np.linspace(0.85, 0.95, 20),
        ...     sigma=np.linspace(0.2, 1.0, 20),
        ...     L_0=1.2,
        ...     delta=-0.25,
        ... )
        """
        import matplotlib.pyplot as plt

        def is_iterable(x):
            return isinstance(x, (list, np.ndarray))

        # Classify parameters as sweep or fixed, preserving order
        param_info = [
            ('p_c', p_c, self.p_),
            ('sigma', sigma, r'$\sigma$'),
            ('L_0', L_0, r'$L_0$'),
            ('delta', delta, r'$\Delta$'),
        ]
        sweep_params = []
        fixed_params = {}
        for name, val, label in param_info:
            if is_iterable(val):
                sweep_params.append((name, np.asarray(val), label))
            else:
                fixed_params[name] = val

        if len(sweep_params) != 2:
            raise NotImplementedError(
                f"Exactly 2 parameter arrays required for pcolormesh, got {len(sweep_params)}. "
                "got {len(sweep_params)}, need exactly 2 of the 4 BKT parameters (p_c, sigma, L_0, delta)."
            )

        # Extract sweep arrays (order preserved from signature)
        name1, arr1, label1 = sweep_params[0]
        name2, arr2, label2 = sweep_params[1]

        # Auto-derive ranges from sweep arrays so lmfit doesn't clamp values
        def _auto_range(val, default):
            if isinstance(val, (list, np.ndarray)):
                arr = np.asarray(val)
                return (float(arr.min()), float(arr.max()))
            return default

        if p_c_range is None:
            p_c_range = _auto_range(p_c, (0, 1))
        if sigma_range is None:
            sigma_range = _auto_range(sigma, (0.1, 5))
        if L_0_range is None:
            L_0_range = _auto_range(L_0, (0.01, float(np.min(self.L_i) - 1e-6)))
        if delta_range is None:
            delta_range = _auto_range(delta, (-2, 2))

        # Define chi2 computation function
        def compute_chi2(i, j):
            params = dict(fixed_params)
            params[name1] = arr1[i]
            params[name2] = arr2[j]
            try:
                res = self.datacollapse_bkt(
                    p_c=params['p_c'], sigma=params['sigma'], L_0=params['L_0'], delta=params['delta'],
                    p_c_vary=False, sigma_vary=False, L_0_vary=False, delta_vary=False,
                    p_c_range=p_c_range,
                    sigma_range=sigma_range, L_0_range=L_0_range, delta_range=delta_range,
                )
                return i, j, res.redchi
            except Exception:
                return i, j, np.nan

        # Generate all tasks
        tasks = [(i, j) for i in range(len(arr1)) for j in range(len(arr2))]

        # Execute
        if n_jobs == 1:
            results = [compute_chi2(i, j) for i, j in tasks]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(compute_chi2)(i, j) for i, j in tasks
            )

        # Assemble grid
        chi2_grid = np.zeros((len(arr1), len(arr2)))
        for i, j, val in results:
            chi2_grid[i, j] = val

        def _trim_nan_edges(values: np.ndarray, grid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
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

        plot_arr1, plot_grid = _trim_nan_edges(arr1, chi2_grid, axis=0)
        plot_arr2, plot_grid = _trim_nan_edges(arr2, plot_grid, axis=1)

        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        def _edges_from_centers(values: np.ndarray) -> np.ndarray:
            if values.size == 1:
                return np.array([values[0] - 0.5, values[0] + 0.5])
            deltas = np.diff(values)
            first = values[0] - deltas[0] / 2
            last = values[-1] + deltas[-1] / 2
            mids = values[:-1] + deltas / 2
            return np.concatenate([[first], mids, [last]])

        # Plot pcolormesh
        im = ax.pcolormesh(
            plot_arr2, plot_arr1, np.log(plot_grid) if log_chi2 else plot_grid,
            cmap=cmap,
            shading='auto',
        )
        ax.margins(x=0, y=0)
        x_edges = _edges_from_centers(plot_arr2)
        y_edges = _edges_from_centers(plot_arr1)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

        # Set axis labels (first sweep param → ylabel, second → xlabel)
        ax.set_ylabel(_format_token(label1))
        ax.set_xlabel(_format_token(label2))

        # Add colorbar
        cb_label = r'$\log(\chi_\nu^2)$' if log_chi2 else r'$\chi_\nu^2$'
        if colorbar_position == 'right':
            plt.colorbar(im, ax=ax, label=cb_label)
        elif colorbar_position == 'top':
            axins = ax.inset_axes([0.4, 1.02, 0.6, 0.05])
            plt.colorbar(im, cax=axins, label='', orientation='horizontal')
            axins.xaxis.tick_top()
            axins.text(0.4, 1.02, cb_label, ha='right', va='bottom', transform=ax.transAxes)

        # Find minimum and plot marker
        idx1, idx2 = np.unravel_index(np.nanargmin(plot_grid), plot_grid.shape)
        ax.plot(plot_arr2[idx2], plot_arr1[idx1], marker='x', color='red', markersize=10)
        ax.set_title(f'min $\\chi^2_\\nu$ = {plot_grid[idx1, idx2]:.1f}')

        # Draw contour at 1.3× minimum
        pts = ax.contour(
            plot_arr2, plot_arr1, plot_grid,
            levels=[1.3 * plot_grid[idx1, idx2]],
            colors='y'
        )

        # Re-apply bounds to avoid autoscale margins after contour/marker
        ax.margins(x=0, y=0)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_autoscale_on(False)

        # Select contour with most vertices (longest)
        paths = pts.get_paths()
        if paths:
            longest_path = max(paths, key=lambda p: len(p.vertices))
            vertices = longest_path.vertices
            error1 = (vertices[:, 1].min(), vertices[:, 1].max())
            error2 = (vertices[:, 0].min(), vertices[:, 0].max())
        else:
            error1 = (np.nan, np.nan)
            error2 = (np.nan, np.nan)

        # Build result dict
        result = {
            'p_c': None,
            'sigma': None,
            'L_0': None,
            'delta': None,
            'p_c_error': None,
            'sigma_error': None,
            'L_0_error': None,
            'delta_error': None,
            'chi2_grid': chi2_grid,
        }

        # Set optimal values
        result[name1] = plot_arr1[idx1]
        result[name2] = plot_arr2[idx2]
        for fname, fval in fixed_params.items():
            result[fname] = fval

        # Set errors for swept params
        result[f'{name1}_error'] = error1
        result[f'{name2}_error'] = error2

        # Print summary
        print(f'{name1}={arr1[idx1]:.4f}, {name2}={arr2[idx2]:.4f}, chi2={chi2_grid[idx1, idx2]:.4f}')
        print(f'{name1} error=({error1[0]:.4f}, {error1[1]:.4f}), {name2} error=({error2[0]:.4f}, {error2[1]:.4f})')

        return result


def grid_search(
    n1_list: list[int],
    n2_list: list[int],
    p_c: float,
    nu: float,
    y: float,
    p_c_range: tuple[float, float],
    nu_range: tuple[float, float],
    verbose: bool = False,
    **kwargs,
) -> dict[tuple[int, int], DataCollapse]:
    """Grid search over polynomial orders (n1, n2) for optimal model selection.

    Fits `datacollapse_with_drift_GLS` for each combination of (n1, n2) and
    returns a dictionary of fitted models for comparison.

    Parameters
    ----------
    n1_list : list[int]
        List of n1 values (relevant scaling polynomial order) to search.
    n2_list : list[int]
        List of n2 values (irrelevant scaling polynomial order) to search.
    p_c : float
        Initial guess for critical point.
    nu : float
        Initial guess for correlation length exponent.
    y : float
        Initial guess for irrelevant scaling exponent.
    p_c_range : tuple[float, float]
        Bounds (min, max) for p_c.
    nu_range : tuple[float, float]
        Bounds (min, max) for nu.
    verbose : bool, optional
        If True, print (n1, n2) during fitting.
    **kwargs
        Arguments passed to DataCollapse constructor (df, p_, L_, params, etc.).

    Returns
    -------
    dict[tuple[int, int], DataCollapse]
        Dictionary mapping (n1, n2) to fitted DataCollapse objects.
    """
    # red_chi2_list=np.zeros((len(n1_list),len(n2_list)))
    from tqdm import tqdm
    model_dict={}

    n_list=[(n1,n2) for n1 in n1_list for n2 in n2_list]
    for (n1,n2) in tqdm(n_list):
        if verbose:
            print(n1,n2)
        dc=DataCollapse(**kwargs)
        try:
            res0=dc.datacollapse_with_drift_GLS(n1=n1,n2=n2,p_c=p_c,nu=nu,y=y,p_c_range=p_c_range,nu_range=nu_range,)
        except:
            print(f'Fitting Failed for (n1={n1},n2={n2})')
        model_dict[(n1,n2)]=dc

    return model_dict

def plot_chi2_ratio(
    model_dict: dict[tuple[int, int], DataCollapse],
    L1: bool = False,
) -> None:
    """Plot reduced chi-squared and irrelevant contribution ratio for model selection.

    Creates a dual-axis plot showing reduced chi-squared (left axis) and
    the ratio of irrelevant to total variance (right axis) as functions of n1.

    Parameters
    ----------
    model_dict : dict[tuple[int, int], DataCollapse]
        Dictionary from `grid_search()` mapping (n1, n2) to fitted models.
    L1 : bool, optional
        If True, use L1 norm for irrelevant ratio. If False (default),
        use variance-based ratio (ESS_irr / TSS).
    """
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    color_list=['r','b','c','m','y','k','g']

    n1_list=[]
    n2_list=[]
    for key in model_dict.keys():
        if key[0] not in n1_list:
            n1_list.append(key[0])
        if key[1] not in n2_list:
            n2_list.append(key[1])

    for n2 in n2_list:
        ax.plot(n1_list,[(model_dict[n1,n2].res.redchi if hasattr(model_dict[n1,n2],"res") else np.nan) for n1 in n1_list],label=f'$n_2$={n2}',color=color_list[n2],marker='.')
        
    ax.set_yscale('log')
    ax.axhline(1,color='k',ls='dotted',lw=0.5)
    ax.fill_between(n1_list,0.5,5,alpha=0.5,color='cyan')
    ax.legend()

    ax2=ax.twinx()
    
    for n2 in n2_list:
        if L1:
            ratio=[np.abs(model_dict[n1,n2].y_i_irrelevant/model_dict[n1,n2].y_i_minus_irrelevant).mean() if hasattr(model_dict[n1,n2],"res") else np.nan for n1 in n1_list]
        else:
            # ratio=[np.var(model_dict[n1,n2].y_i_irrelevant)/np.var(model_dict[n1,n2].y_i) if hasattr(model_dict[n1,n2],"res") else np.nan for n1 in n1_list]
            ratio=[]
            for n1 in n1_list:
                if hasattr(model_dict[n1,n2],"res"):
                    y_i_irrelevant_mean=np.sum(model_dict[n1,n2].y_i_irrelevant/model_dict[n1,n2].d_i**2)/np.sum(1/model_dict[n1,n2].d_i**2)
                    y_i_mean=np.sum(model_dict[n1,n2].y_i/model_dict[n1,n2].d_i**2)/np.sum(1/model_dict[n1,n2].d_i**2)
                    ESS_irr=np.sum((model_dict[n1,n2].y_i_irrelevant-y_i_irrelevant_mean)**2/model_dict[n1,n2].d_i**2)
                    TSS=np.sum((model_dict[n1,n2].y_i-y_i_mean)**2/model_dict[n1,n2].d_i**2)
                    ratio.append(ESS_irr/TSS)
                else:
                    ratio.append(np.nan)
        ax2.plot(n1_list,ratio,label=f'$n_2$={n2}',color=color_list[n2],ls='--',marker='.')
        ax2.set_ylim([0,1.05])

    ax.set_xlabel('$n_1$')
    ax.set_ylabel(r'$\chi_{\nu}^2$')
    ax2.set_ylabel('Irrelevant contribution')
    ax2.fill_between(n1_list,0.,0.1,alpha=0.2,color='orange')


def extrapolate_fitting(
    data: dict[Any, pd.DataFrame],
    params: dict[str, Any],
    p_range: list[float],
    p_: str,
    L_: str,
    Lmin: int = 12,
    Lmax: int = 24,
    nu: float = 1.3,
    p_c: float = 0.5,
    threshold: tuple[float, float] = (-1, 1),
) -> dict[Any, DataCollapse]:
    """Fit data collapse across different threshold values (deprecated/unused)."""
    from tqdm import tqdm
    dc={}
    for key,val in tqdm(data.items()):
        if threshold[0]<key<threshold[1]:
            dc[key]=DataCollapse(df=val,params=params,Lmin=Lmin,Lmax=Lmax,p_range=p_range,p_=p_,L_=L_)
            dc[key].datacollapse(nu=nu,p_c=p_c,)
    return dc


def plot_extrapolate_fitting(
    dc: dict[Any, DataCollapse],
    ax: Axes | None = None,
) -> None:
    """Plot extrapolate fitting results (deprecated/unused)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig,ax=plt.subplots()
    xlist=list(dc.keys())
    nu=[dc[key].res.params['nu'].value for key in dc.keys()]
    nu_err=[dc[key].res.params['nu'].stderr for key in dc.keys()]
    ax.errorbar(xlist,nu,yerr=nu_err,fmt='.-',capsize=3,label=r'$\nu$',color='k')
    ax2=ax.twinx()
    p_c=[dc[key].res.params['p_c'].value for key in dc.keys()]
    p_c_err=[dc[key].res.params['p_c'].stderr for key in dc.keys()]
    ax2.errorbar(xlist,p_c,yerr=p_c_err,fmt='.-',capsize=3,label='$p_c$',color='b')

    ax2.tick_params(axis='y', labelcolor='b')

    # ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Threshold of SV')
    ax.set_ylabel(r'$\nu$',color='k')
    ax2.set_ylabel(r'$p_c$',color='b')

class optimal_df:
    """Container for storing optimal fitting results across different parameter sets."""

    def __init__(self, names: list[str] | None = None) -> None:
        """Initialize storage for optimal fitting results.

        Parameters
        ----------
        names : list[str] | None, optional
            Index level names for organizing results, by default
            ['Metrics', 'p_proj', 'p_ctrl'].
        """
        import pandas as pd
        if names is None:
            names = ['Metrics', 'p_proj', 'p_ctrl']
        self.names=names
        self.opt_df=pd.DataFrame(
                columns=['p_c', 'p_c_error', 'nu', 'nu_error', 'y', 'y_error'],
                index= pd.MultiIndex(levels=[[]]*len(names), codes=[[]]*len(names), names=names)
            )

    def add_optimal(self, model: DataCollapse) -> None:
        """Add fitting results from a DataCollapse model to storage.

        Extracts p_c, nu, y and their errors from the fitted model and
        appends them to `self.opt_df`.

        Parameters
        ----------
        model : DataCollapse
            Fitted DataCollapse object with `res` and `params` attributes.
        """
        import pandas as pd
        df_new = pd.DataFrame([model])
        p_c_key=frozenset(self.names)-frozenset(model.params.keys())

        index_list=[]
        for name in self.names:
            if name in model.params:
                index_list.append(model.params[name])
            else:
                index_list.append(None)
        index = pd.MultiIndex.from_tuples([tuple(index_list)],names=self.names)
        p_c=model.res.params['p_c'].value
        p_c_error=model.res.params['p_c'].stderr
        nu=model.res.params['nu'].value
        nu_error=model.res.params['nu'].stderr
        if 'y' in model.res.params:
            y=model.res.params['y'].value
            y_error=model.res.params['y'].stderr
        else:
            y=None
            y_error=None
        new={
            'p_c':p_c,
            'p_c_error':p_c_error,
            'nu': nu,
            'nu_error': nu_error,
            'y': y,
            'y_error': y_error}
        new_df=pd.DataFrame(new,index=index)
        self.opt_df=pd.concat([self.opt_df,new_df],axis=0)
    
    # def delete_from_last(self,loc):
    #     total=np.arange(len(self.opt_df))
    #     self.opt_df=self.opt_df.iloc([i for i in total if i not in loc])
        
def bootstrapping(
    df: pd.DataFrame,
    params: dict[str, Any],
    p_: str,
    L_: str,
    p_range: list[float],
    nu: float,
    p_c: float,
    rng: int | np.random.Generator = 0,
    Lmin: int | None = None,
    Lmax: int | None = None,
    size: int | None = None,
    replace: bool = True,
    method: str = 'leastsq',
    p_c_vary: bool = True,
    nu_range: tuple[float, float] = (0.5, 2),
    **kwargs,
) -> DataCollapse:
    """Perform bootstrap resampling for error estimation.

    Creates a resampled dataset by randomly sampling observations with
    replacement, then fits a DataCollapse model. Repeat multiple times
    to estimate parameter uncertainties.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with MultiIndex and 'observations' column.
    params : dict[str, Any]
        Fixed parameter values to select from the DataFrame.
    p_ : str
        Name of the tuning parameter index level.
    L_ : str
        Name of the system size index level.
    p_range : list[float]
        Range [min, max] of tuning parameter to include.
    nu : float
        Initial guess for correlation length exponent.
    p_c : float
        Initial guess for critical point.
    rng : int | np.random.Generator, optional
        Random seed or generator for reproducibility.
    Lmin : int | None, optional
        Minimum system size to include.
    Lmax : int | None, optional
        Maximum system size to include.
    size : int | None, optional
        Number of samples to draw per observation. If None, use original size.
    replace : bool, optional
        Whether to sample with replacement, by default True.
    method : str, optional
        Optimization method for lmfit, by default 'leastsq'.
    p_c_vary : bool, optional
        Whether to fit p_c, by default True.
    nu_range : tuple[float, float], optional
        Bounds (min, max) for nu, by default (0.5, 2).
    **kwargs
        Additional arguments passed to DataCollapse constructor.

    Returns
    -------
    DataCollapse
        Fitted DataCollapse object on resampled data.
    """
    rng=np.random.default_rng(rng)
    df_small=df.xs(params.values(),level=list(params.keys()),drop_level=False)
    df_resample=df_small.applymap(lambda x: rng.choice(x,size=len(x) if size is None else min(size,len(x)),replace=replace))
    dc=DataCollapse(df=df_resample,params=params,Lmin=Lmin,Lmax=Lmax,p_range=p_range,p_=p_,L_=L_,**kwargs)
    dc.datacollapse(nu=nu,p_c=p_c,method=method,p_c_vary=p_c_vary,nu_range=nu_range)
    return dc
